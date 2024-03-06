#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <time.h>
#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
using namespace std;

#define CUDA_CHECK()                                                                                   \
    if ((cudaPeekAtLastError()) != cudaSuccess) {                                                      \
        printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__ - 1); \
        exit(-1);                                                                                      \
    }

constexpr int kWarpSize = 32;

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
    if (threadIdx.x == 0) {
        result_broadcast = result;
    }
    __syncthreads();
    return result_broadcast;
}

template <typename T>
__inline__ __device__ T Inf();

template <>
__inline__ __device__ float Inf<float>() {
    return CUDART_INF_F;
}

template <>
__inline__ __device__ double Inf<double>() {
    return CUDART_INF;
}

template <typename T>
__inline__ __device__ T Exp(T x);

template <>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
    return __expf(x);
#else
    return exp(x);
#endif
}

template <>
__inline__ __device__ double Exp<double>(double x) {
    return exp(x);
}

template <typename T>
__inline__ __device__ T Div(T a, T b);

template <>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
    return __fdividef(a, b);
#else
    return a / b;
#endif
}

template <>
__inline__ __device__ double Div<double>(double a, double b) {
    return a / b;
}

template <typename T>
__inline__ __device__ T Log(T x);

template <>
__inline__ __device__ float Log<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
    return __logf(x);
#else
    return log(x);
#endif
}
template <>
__inline__ __device__ double Log<double>(double x) {
    return log(x);
}

inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves, int* num_blocks) {
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
    return cudaSuccess;
}

template <typename T>
struct DefaultComputeType {
    using type = T;
};

template <>
struct DefaultComputeType<half> {
    using type = float;
};

template <typename T, int N>
struct GetPackType {
    using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template <typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int N>
union Pack {
    static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
    __device__ Pack() {}
    PackType<T, N> storage;
    T elem[N];
};

template <typename SRC, typename DST>
struct DirectLoad {
    DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
    template <int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) const {
        Pack<SRC, N> pack;
        const int64_t offset = (row * row_size + col) / N;
        pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            dst[i] = static_cast<DST>(pack.elem[i]);
        }
    }
    const SRC* src;
    int64_t row_size;
};

template <typename SRC, typename DST>
struct DirectStore {
    DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
    template <int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> pack;
        const int64_t offset = (row * row_size + col) / N;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            pack.elem[i] = static_cast<DST>(src[i]);
        }
        *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
    }
    DST* dst;
    int64_t row_size;
};

enum class Algorithm {
    kSoftmax = 0,
    kLogSoftmax = 1,
};

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          bool padding,
          Algorithm algorithm>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    static_assert(cols_per_thread % pack_size == 0, "");
    static_assert(thread_group_width <= kWarpSize, "");
    static_assert(kWarpSize % thread_group_width == 0, "");
    constexpr int num_packs = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread * thread_group_width);
    ComputeType buf[rows_per_access][cols_per_thread];

    const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int num_global_thread_group = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group * rows_per_access;
    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
        ComputeType thread_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_max[row_id] = -Inf<ComputeType>();
            ComputeType* row_buf = buf[row_id];
#pragma unroll
            for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
                    for (int i = 0; i < pack_size; ++i) {
                        thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < pack_size; ++i) {
                        row_buf[pack_offset + i] = -Inf<ComputeType>();
                    }
                }
            }
        }
        ComputeType warp_max[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
        }
        ComputeType thread_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_sum[row_id] = 0;
            ComputeType* row_buf = buf[row_id];
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
                    thread_sum[row_id] += row_buf[i];
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    row_buf[i] -= warp_max[row_id];
                    thread_sum[row_id] += Exp(row_buf[i]);
                } else {
                    __trap();
                }
            }
        }
        ComputeType warp_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
        }
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            ComputeType* row_buf = buf[row_id];
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    row_buf[i] -= Log(warp_sum[row_id]);
                } else {
                    __trap();
                }
            }
#pragma unroll
            for (int i = 0; i < num_packs; ++i) {
                const int col = (i * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
                }
            }
        }
    }
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          bool padding,
          Algorithm algorithm>
inline cudaError_t LaunchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    constexpr int block_size = 128;
    constexpr int waves = 32;
    static_assert(block_size % thread_group_width == 0, "");
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    const int64_t num_blocks = (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
        if (err != cudaSuccess) {
            return err;
        }
    }
    SoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width, rows_per_access, padding, algorithm>
        <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    if (cols == cols_per_thread * thread_group_width) {
        return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width, rows_per_access, false, algorithm>(
            stream, load, store, rows, cols);
    } else {
        return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width, rows_per_access, true, algorithm>(
            stream, load, store, rows, cols);
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxWarpImplCols(cudaStream_t stream,
                                                                                       LOAD load,
                                                                                       STORE store,
                                                                                       const int64_t rows,
                                                                                       const int64_t cols) {
    if (cols <= 0) {
        return cudaErrorInvalidValue;
    }
#define DEFINE_ONE_ELIF(thread_group_width)                                                                                          \
    else if (cols <= (thread_group_width) * pack_size) {                                                                             \
        if (rows % 2 == 0) {                                                                                                         \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, thread_group_width, 2, algorithm>( \
                stream, load, store, rows, cols);                                                                                    \
        } else {                                                                                                                     \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, thread_group_width, 1, algorithm>( \
                stream, load, store, rows, cols);                                                                                    \
        }                                                                                                                            \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                                                                       \
    else if (cols <= (col) * kWarpSize) {                                                                                                          \
        return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, algorithm>(stream, load, store, rows, cols); \
    }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(7)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(9)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(11)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(13)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(15)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(17)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(19)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(21)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(23)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(25)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(27)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(29)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(31)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else {
        return cudaErrorInvalidValue;
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxWarpImplCols(cudaStream_t stream,
                                                                                       LOAD load,
                                                                                       STORE store,
                                                                                       const int64_t rows,
                                                                                       const int64_t cols) {
    if (cols <= 0) {
        return cudaErrorInvalidValue;
    }
#define DEFINE_ONE_ELIF(thread_group_width)                                                                                          \
    else if (cols <= (thread_group_width) * pack_size) {                                                                             \
        if (rows % 2 == 0) {                                                                                                         \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, thread_group_width, 2, algorithm>( \
                stream, load, store, rows, cols);                                                                                    \
        } else {                                                                                                                     \
            return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, thread_group_width, 1, algorithm>( \
                stream, load, store, rows, cols);                                                                                    \
        }                                                                                                                            \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                                                                       \
    else if (cols <= (col) * kWarpSize) {                                                                                                          \
        return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, algorithm>(stream, load, store, rows, cols); \
    }
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else {
        return cudaErrorInvalidValue;
    }
}

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxWarpImplPackSize {
    cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
        if (cols % 2 == 0) {
            return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, algorithm>(stream, load, store, rows, cols);
        } else {
            return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, algorithm>(stream, load, store, rows, cols);
        }
    }
};

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    return DispatchSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType, algorithm>()(stream, load, store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        ComputeType thread_max = -Inf<ComputeType>();
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                buf[i * num_packs + pack_id] = pack[i];
                thread_max = max(thread_max, pack[i]);
            }
        }
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
        ComputeType thread_sum = 0;
        for (int col = tid; col < cols; col += block_size) {
            if (algorithm == Algorithm::kSoftmax) {
                const ComputeType exp_x = Exp(buf[col] - row_max);
                buf[col] = exp_x;
                thread_sum += exp_x;
            } else {
                const ComputeType x = buf[col] - row_max;
                buf[col] = x;
                thread_sum += Exp(x);
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    pack[i] = buf[i * num_packs + pack_id] - Log(row_sum);
                } else {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, int smem, const int64_t rows, const int64_t cols) {
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) {
            return err;
        }
    }
    SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream,
                                                            LOAD load,
                                                            STORE store,
                                                            const int64_t rows,
                                                            const int64_t cols,
                                                            bool* success) {
    constexpr int block_size_conf_1 = 128;
    constexpr int block_size_conf_2 = 256;
    constexpr int block_size_conf_3 = 512;
    constexpr int block_size_conf_4 = 1024;
    const size_t smem = cols * sizeof(ComputeType);
    int max_active_blocks_conf_1;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_1, SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>, block_size_conf_1,
            smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_1 <= 0) {
        *success = false;
        return cudaSuccess;
    }
    int max_active_blocks_conf_4;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_4, SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>, block_size_conf_4,
            smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>(stream, load, store, smem, rows, cols);
    }
    int max_active_blocks_conf_3;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_3, SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>, block_size_conf_3,
            smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>(stream, load, store, smem, rows, cols);
    }
    int max_active_blocks_conf_2;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_2, SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>, block_size_conf_2,
            smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>(stream, load, store, smem, rows, cols);
    }
    *success = true;
    return LaunchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>(stream, load, store, smem, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct TryDispatchSoftmaxBlockSMemImplPackSize {
    cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols, bool* success) {
        if (cols % 2 == 0) {
            return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, algorithm>(stream, load, store, rows, cols, success);
        } else {
            return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, algorithm>(stream, load, store, rows, cols, success);
        }
    }
};

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream,
                                                   LOAD load,
                                                   STORE store,
                                                   const int64_t rows,
                                                   const int64_t cols,
                                                   bool* success) {
    return TryDispatchSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType, algorithm>()(stream, load, store, rows, cols, success);
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        ComputeType thread_max = -Inf<ComputeType>();
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                thread_max = max(thread_max, pack[i]);
            }
        }
        const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
        ComputeType thread_sum = 0;
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                thread_sum += Exp(pack[i] - row_max);
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
            load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    pack[i] = Div(Exp(pack[i] - row_max), row_sum);
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    pack[i] = (pack[i] - row_max) - Log(row_sum);
                } else {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    constexpr int block_size = 1024;
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) {
            return err;
        }
    }
    SoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxBlockUncachedImplPackSize {
    cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
        if (cols % 2 == 0) {
            return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 2, algorithm>(stream, load, store, rows, cols);
        } else {
            return LaunchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 1, algorithm>(stream, load, store, rows, cols);
        }
    }
};

template <typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
    return DispatchSoftmaxBlockUncachedImplPackSize<LOAD, STORE, ComputeType, algorithm>()(stream, load, store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchSoftmax(cudaStream_t stream,
                                                                                                             LOAD load,
                                                                                                             STORE store,
                                                                                                             const int64_t rows,
                                                                                                             const int64_t cols) {
    if (cols < 1024) {
        return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(stream, load, store, rows, cols);
    } else {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err = TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(stream, load, store, rows, cols,
                                                                                                             &dispatch_smem_impl_success);
            if (err != cudaSuccess) {
                return err;
            }
        }
        if (!dispatch_smem_impl_success) {
            return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(stream, load, store, rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchSoftmax(cudaStream_t stream,
                                                                                                            LOAD load,
                                                                                                            STORE store,
                                                                                                            const int64_t rows,
                                                                                                            const int64_t cols) {
    return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kSoftmax>(stream, load, store, rows, cols);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchLogSoftmax(cudaStream_t stream,
                                                                                                                LOAD load,
                                                                                                                STORE store,
                                                                                                                const int64_t rows,
                                                                                                                const int64_t cols) {
    if (cols <= 1024) {
        return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(stream, load, store, rows, cols);
    } else {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err = TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(stream, load, store, rows, cols,
                                                                                                                &dispatch_smem_impl_success);
            if (err != cudaSuccess) {
                return err;
            }
        }
        if (!dispatch_smem_impl_success) {
            return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(stream, load, store, rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchLogSoftmax(cudaStream_t stream,
                                                                                                               LOAD load,
                                                                                                               STORE store,
                                                                                                               const int64_t rows,
                                                                                                               const int64_t cols) {
    return DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, Algorithm::kLogSoftmax>(stream, load, store, rows, cols);
}

template <typename LOAD_Y,
          typename LOAD_DY,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          bool padding,
          Algorithm algorithm>
__global__ void SoftmaxGradWarpImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows, const int64_t cols) {
    static_assert(cols_per_thread % pack_size == 0, "");
    constexpr int pack_per_thread = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread * thread_group_width);
    static_assert(thread_group_width <= kWarpSize, "");
    static_assert(kWarpSize % thread_group_width == 0, "");
    ComputeType y_buf[rows_per_access][cols_per_thread];
    ComputeType dy_buf[rows_per_access][cols_per_thread];
    const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int num_global_thread_group = gridDim.x * blockDim.y;
    const int lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group * rows_per_access;
    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
        ComputeType thread_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_sum[row_id] = 0;
            ComputeType* row_y_buf = y_buf[row_id];
            ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
            for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    load_y.template load<pack_size>(row_y_buf + pack_offset, row + row_id, col);
                    load_dy.template load<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
#pragma unroll
                    for (int i = 0; i < pack_size; ++i) {
                        if (algorithm == Algorithm::kSoftmax) {
                            thread_sum[row_id] += row_y_buf[pack_offset + i] * row_dy_buf[pack_offset + i];
                        } else if (algorithm == Algorithm::kLogSoftmax) {
                            thread_sum[row_id] += row_dy_buf[pack_offset + i];
                        } else {
                            __trap();
                        }
                    }
                }
            }
        }
        ComputeType warp_sum[rows_per_access];
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
        }
#pragma unroll
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            ComputeType* row_y_buf = y_buf[row_id];
            ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
            for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                    for (int i = 0; i < pack_size; ++i) {
                        if (algorithm == Algorithm::kSoftmax) {
                            row_dy_buf[pack_offset + i] = (row_dy_buf[pack_offset + i] - warp_sum[row_id]) * row_y_buf[pack_offset + i];
                        } else if (algorithm == Algorithm::kLogSoftmax) {
                            row_dy_buf[pack_offset + i] -= Exp(row_y_buf[pack_offset + i]) * warp_sum[row_id];
                        } else {
                            __trap();
                        }
                    }
                    store.template store<pack_size>(row_dy_buf + pack_offset, row + row_id, col);
                }
            }
        }
    }
}

template <typename LOAD_Y,
          typename LOAD_DY,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          bool padding,
          Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradWarpImpl(cudaStream_t stream,
                                             LOAD_Y load_y,
                                             LOAD_DY load_dy,
                                             STORE store,
                                             const int64_t rows,
                                             const int64_t cols) {
    constexpr int block_size = 128;
    constexpr int waves = 32;
    static_assert(block_size % thread_group_width == 0, "");
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    const int64_t num_blocks = (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
        if (err != cudaSuccess) {
            return err;
        }
    }
    SoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width, rows_per_access, padding, algorithm>
        <<<grid_dim_x, block_dim, 0, stream>>>(load_y, load_dy, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD_Y,
          typename LOAD_DY,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int cols_per_thread,
          int thread_group_width,
          int rows_per_access,
          Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImplPadding(cudaStream_t stream,
                                                      LOAD_Y load_y,
                                                      LOAD_DY load_dy,
                                                      STORE store,
                                                      const int64_t rows,
                                                      const int64_t cols) {
    if (cols == cols_per_thread * thread_group_width) {
        return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width, rows_per_access, false,
                                         algorithm>(stream, load_y, load_dy, store, rows, cols);
    } else {
        return LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width, rows_per_access, true,
                                         algorithm>(stream, load_y, load_dy, store, rows, cols);
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(cudaStream_t stream,
                                                                                           LOAD_Y load_y,
                                                                                           LOAD_DY load_dy,
                                                                                           STORE store,
                                                                                           const int64_t rows,
                                                                                           const int64_t cols) {
    if (cols <= 0) {
        return cudaErrorInvalidValue;
    }
#define DEFINE_ONE_ELIF(thread_group_width)                                                                                                         \
    else if (cols <= (thread_group_width) * pack_size) {                                                                                            \
        if (rows % 2 == 0) {                                                                                                                        \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, pack_size, thread_group_width, 2, algorithm>( \
                stream, load_y, load_dy, store, rows, cols);                                                                                        \
        } else {                                                                                                                                    \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, pack_size, thread_group_width, 1, algorithm>( \
                stream, load_y, load_dy, store, rows, cols);                                                                                        \
        }                                                                                                                                           \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                                                     \
    else if (cols <= (col) * kWarpSize) {                                                                                        \
        return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, kWarpSize, 1, algorithm>( \
            stream, load_y, load_dy, store, rows, cols);                                                                         \
    }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(7)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(9)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(11)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(13)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(15)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(17)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(19)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(21)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(23)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(25)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(27)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(29)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(31)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else {
        return cudaErrorInvalidValue;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchSoftmaxGradWarpImplCols(cudaStream_t stream,
                                                                                           LOAD_Y load_y,
                                                                                           LOAD_DY load_dy,
                                                                                           STORE store,
                                                                                           const int64_t rows,
                                                                                           const int64_t cols) {
    if (cols <= 0) {
        return cudaErrorInvalidValue;
    }
#define DEFINE_ONE_ELIF(thread_group_width)                                                                                                         \
    else if (cols <= (thread_group_width) * pack_size) {                                                                                            \
        if (rows % 2 == 0) {                                                                                                                        \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, pack_size, thread_group_width, 2, algorithm>( \
                stream, load_y, load_dy, store, rows, cols);                                                                                        \
        } else {                                                                                                                                    \
            return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, pack_size, thread_group_width, 1, algorithm>( \
                stream, load_y, load_dy, store, rows, cols);                                                                                        \
        }                                                                                                                                           \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                                                     \
    else if (cols <= (col) * kWarpSize) {                                                                                        \
        return DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col, kWarpSize, 1, algorithm>( \
            stream, load_y, load_dy, store, rows, cols);                                                                         \
    }
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(6)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(10)
    DEFINE_ONE_ELIF(12)
    DEFINE_ONE_ELIF(14)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(18)
    DEFINE_ONE_ELIF(20)
    DEFINE_ONE_ELIF(22)
    DEFINE_ONE_ELIF(24)
    DEFINE_ONE_ELIF(26)
    DEFINE_ONE_ELIF(28)
    DEFINE_ONE_ELIF(30)
    DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
    else {
        return cudaErrorInvalidValue;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxGradWarpImplPackSize {
    cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows, const int64_t cols) {
        if (cols % 2 == 0) {
            return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(stream, load_y, load_dy, store, rows, cols);
        } else {
            return DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(stream, load_y, load_dy, store, rows, cols);
        }
    }
};

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradWarpImpl(cudaStream_t stream,
                                               LOAD_Y load_y,
                                               LOAD_DY load_dy,
                                               STORE store,
                                               const int64_t rows,
                                               const int64_t cols) {
    return DispatchSoftmaxGradWarpImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType, algorithm>()(stream, load_y, load_dy, store, rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockSMemImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows, const int64_t cols) {
    extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
    auto* y_buf = reinterpret_cast<ComputeType*>(grad_shared_buf);
    auto* dy_buf = y_buf + cols;
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        ComputeType thread_sum = 0;
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType y_pack[pack_size];
            ComputeType dy_pack[pack_size];
            load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
            load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                y_buf[i * num_packs + pack_id] = y_pack[i];
                dy_buf[i * num_packs + pack_id] = dy_pack[i];
                if (algorithm == Algorithm::kSoftmax) {
                    thread_sum += y_pack[i] * dy_pack[i];
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    thread_sum += dy_pack[i];
                } else {
                    __trap();
                }
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType pack[pack_size];
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    pack[i] = (dy_buf[i * num_packs + pack_id] - row_sum) * y_buf[i * num_packs + pack_id];
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    pack[i] = dy_buf[i * num_packs + pack_id] - Exp(y_buf[i * num_packs + pack_id]) * row_sum;
                } else {
                    __trap();
                }
            }
            store.template store<pack_size>(pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
inline cudaError_t
LaunchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, int smem, const int64_t rows, const int64_t cols) {
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) {
            return err;
        }
    }
    SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, smem, stream>>>(load_y, load_dy, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImplBlockSize(cudaStream_t stream,
                                                                LOAD_Y load_y,
                                                                LOAD_DY load_dy,
                                                                STORE store,
                                                                const int64_t rows,
                                                                const int64_t cols,
                                                                bool* success) {
    constexpr int block_size_conf_1 = 128;
    constexpr int block_size_conf_2 = 256;
    constexpr int block_size_conf_3 = 512;
    constexpr int block_size_conf_4 = 1024;
    const size_t smem = cols * sizeof(ComputeType) * 2;
    int max_active_blocks_conf_1;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_1, SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>,
            block_size_conf_1, smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_1 <= 0) {
        *success = false;
        return cudaSuccess;
    }
    int max_active_blocks_conf_4;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_4, SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>,
            block_size_conf_4, smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_4, algorithm>(stream, load_y, load_dy,
                                                                                                                            store, smem, rows, cols);
    }
    int max_active_blocks_conf_3;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_3, SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>,
            block_size_conf_3, smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_3, algorithm>(stream, load_y, load_dy,
                                                                                                                            store, smem, rows, cols);
    }
    int max_active_blocks_conf_2;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_conf_2, SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>,
            block_size_conf_2, smem);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
        *success = true;
        return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_2, algorithm>(stream, load_y, load_dy,
                                                                                                                            store, smem, rows, cols);
    }
    *success = true;
    return LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_1, algorithm>(stream, load_y, load_dy,
                                                                                                                        store, smem, rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, Algorithm algorithm>
struct TryDispatchSoftmaxGradBlockSMemImplPackSize {
    cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows, const int64_t cols, bool* success) {
        if (cols % 2 == 0) {
            return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(stream, load_y, load_dy, store,
                                                                                                                   rows, cols, success);
        } else {
            return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(stream, load_y, load_dy, store,
                                                                                                                   rows, cols, success);
        }
    }
};

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t TryDispatchSoftmaxGradBlockSMemImpl(cudaStream_t stream,
                                                       LOAD_Y load_y,
                                                       LOAD_DY load_dy,
                                                       STORE store,
                                                       const int64_t rows,
                                                       const int64_t cols,
                                                       bool* success) {
    return TryDispatchSoftmaxGradBlockSMemImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType, algorithm>()(stream, load_y, load_dy, store, rows, cols,
                                                                                                         success);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size, int block_size, Algorithm algorithm>
__global__ void SoftmaxGradBlockUncachedImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows, const int64_t cols) {
    const int tid = threadIdx.x;
    assert(cols % pack_size == 0);
    const int num_packs = cols / pack_size;
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        ComputeType thread_sum = 0;
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType y_pack[pack_size];
            ComputeType dy_pack[pack_size];
            load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
            load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);

#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    thread_sum += y_pack[i] * dy_pack[i];
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    thread_sum += dy_pack[i];
                } else {
                    __trap();
                }
            }
        }
        const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
        for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
            ComputeType y_pack[pack_size];
            ComputeType dy_pack[pack_size];
            load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
            load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                if (algorithm == Algorithm::kSoftmax) {
                    dy_pack[i] = (dy_pack[i] - row_sum) * y_pack[i];
                } else if (algorithm == Algorithm::kLogSoftmax) {
                    dy_pack[i] -= Exp(y_pack[i]) * row_sum;
                } else {
                    __trap();
                }
            }
            store.template store<pack_size>(dy_pack, row, pack_id * pack_size);
        }
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxGradBlockUncachedImpl(cudaStream_t stream,
                                                      LOAD_Y load_y,
                                                      LOAD_DY load_dy,
                                                      STORE store,
                                                      const int64_t rows,
                                                      const int64_t cols) {
    constexpr int block_size = 1024;
    constexpr int waves = 32;
    int grid_dim_x;
    {
        cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
        if (err != cudaSuccess) {
            return err;
        }
    }
    SoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size, algorithm>
        <<<grid_dim_x, block_size, 0, stream>>>(load_y, load_dy, store, rows, cols);
    return cudaPeekAtLastError();
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxGradBlockUncachedImplPackSize {
    cudaError_t operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows, const int64_t cols) {
        if (cols % 2 == 0 && cols > kWarpSize) {
            return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, algorithm>(stream, load_y, load_dy, store, rows, cols);
        } else {
            return LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, algorithm>(stream, load_y, load_dy, store, rows, cols);
        }
    }
};

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxGradBlockUncachedImpl(cudaStream_t stream,
                                                        LOAD_Y load_y,
                                                        LOAD_DY load_dy,
                                                        STORE store,
                                                        const int64_t rows,
                                                        const int64_t cols) {
    return DispatchSoftmaxGradBlockUncachedImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType, algorithm>()(stream, load_y, load_dy, store, rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchSoftmaxGrad(cudaStream_t stream,
                                                                                                                 LOAD_Y load_y,
                                                                                                                 LOAD_DY load_dy,
                                                                                                                 STORE store,
                                                                                                                 const int64_t rows,
                                                                                                                 const int64_t cols) {
    if (cols <= 1024) {
        return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kSoftmax>(stream, load_y, load_dy, store, rows, cols);
    } else {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kSoftmax>(
                stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess) {
                return err;
            }
        }
        if (!dispatch_smem_impl_success) {
            return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kSoftmax>(stream, load_y, load_dy, store,
                                                                                                                  rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchSoftmaxGrad(cudaStream_t stream,
                                                                                                                LOAD_Y load_y,
                                                                                                                LOAD_DY load_dy,
                                                                                                                STORE store,
                                                                                                                const int64_t rows,
                                                                                                                const int64_t cols) {
    return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kSoftmax>(stream, load_y, load_dy, store, rows, cols);
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchLogSoftmaxGrad(cudaStream_t stream,
                                                                                                                    LOAD_Y load_y,
                                                                                                                    LOAD_DY load_dy,
                                                                                                                    STORE store,
                                                                                                                    const int64_t rows,
                                                                                                                    const int64_t cols) {
    if (cols <= 1024) {
        return DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kLogSoftmax>(stream, load_y, load_dy, store, rows, cols);
    } else {
        bool dispatch_smem_impl_success;
        {
            cudaError_t err = TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kLogSoftmax>(
                stream, load_y, load_dy, store, rows, cols, &dispatch_smem_impl_success);
            if (err != cudaSuccess) {
                return err;
            }
        }
        if (!dispatch_smem_impl_success) {
            return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kLogSoftmax>(stream, load_y, load_dy, store,
                                                                                                                     rows, cols);
        }
        return cudaSuccess;
    }
}

template <typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type DispatchLogSoftmaxGrad(cudaStream_t stream,
                                                                                                                   LOAD_Y load_y,
                                                                                                                   LOAD_DY load_dy,
                                                                                                                   STORE store,
                                                                                                                   const int64_t rows,
                                                                                                                   const int64_t cols) {
    return DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, Algorithm::kLogSoftmax>(stream, load_y, load_dy, store, rows,
                                                                                                             cols);
}

#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__ __host__ __forceinline__
#else
#define OF_DEVICE_FUNC inline
#endif

template <typename T, int N>
class NdIndexOffsetHelper {
   public:
    NdIndexOffsetHelper() = default;

    template <class... Ts>
    OF_DEVICE_FUNC explicit NdIndexOffsetHelper(T d0, Ts... dims) {
        constexpr int n = 1 + sizeof...(dims);
        static_assert(n <= N, "");
        T dims_arr[n] = {d0, static_cast<T>(dims)...};
        InitStrides(dims_arr, n);
    }

    OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims) { InitStrides(dims, N); }

    template <typename U>
    OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims) {
        T dims_arr[N];
        for (int i = 0; i < N; ++i) {
            dims_arr[i] = dims[i];
        }
        InitStrides(dims_arr, N);
    }

    OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims, int n) { InitStrides(dims, n); }

    template <typename U>
    OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims, int n) {
        T dims_arr[N];
        for (int i = 0; i < N; ++i) {
            if (i < n) {
                dims_arr[i] = dims[i];
            }
        }
        InitStrides(dims_arr, n);
    }

    virtual ~NdIndexOffsetHelper() = default;

    OF_DEVICE_FUNC T NdIndexToOffset(const T* index) const {
        T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (int i = 0; i < N; ++i) {
            offset += index[i] * stride_[i];
        }
        return offset;
    }

    OF_DEVICE_FUNC T NdIndexToOffset(const T* index, int n) const {
        assert(n <= N);
        T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (int i = 0; i < N; ++i) {
            if (i < n) {
                offset += index[i] * stride_[i];
            }
        }
        return offset;
    }

    template <class... Ts>
    OF_DEVICE_FUNC T NdIndexToOffset(T d0, Ts... others) const {
        constexpr int n = 1 + sizeof...(others);
        static_assert(n <= N, "");
        T index[n] = {d0, others...};
        T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (int i = 0; i < n - 1; ++i) {
            offset += index[i] * stride_[i];
        }
        if (n == N) {
            offset += index[n - 1];
        } else {
            offset += index[n - 1] * stride_[n - 1];
        }
        return offset;
    }

    OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index) const {
        T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (int i = 0; i < N - 1; ++i) {
            const T idx = remaining / stride_[i];
            index[i] = idx;
            remaining = remaining - idx * stride_[i];
        }
        index[N - 1] = remaining;
    }

    OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index, int n) const {
        assert(n <= N);
        T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (int i = 0; i < N; ++i) {
            if (i < n) {
                const T idx = remaining / stride_[i];
                index[i] = idx;
                remaining = remaining - idx * stride_[i];
            }
        }
    }

    template <class... Ts>
    OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T& d0, Ts&... others) const {
        constexpr int n = 1 + sizeof...(others);
        static_assert(n <= N, "");
        T* index[n] = {&d0, &others...};
        T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (int i = 0; i < n - 1; ++i) {
            const T idx = remaining / stride_[i];
            *index[i] = idx;
            remaining = remaining - idx * stride_[i];
        }
        if (n == N) {
            *index[n - 1] = remaining;
        } else {
            *index[n - 1] = remaining / stride_[n - 1];
        }
    }

    OF_DEVICE_FUNC constexpr int Size() const { return N; }

   protected:
    OF_DEVICE_FUNC void InitStrides(const T* dims, const int n) {
        for (int i = n - 1; i < N; ++i) {
            stride_[i] = 1;
        }
        for (int i = n - 2; i >= 0; --i) {
            stride_[i] = dims[i + 1] * stride_[i + 1];
        }
    }

    T stride_[N];
};

inline void SimplifyBroadcastDims(size_t num_a_dims,
                                  const int64_t* a_dims,
                                  size_t num_b_dims,
                                  const int64_t* b_dims,
                                  size_t* simplified_num_dims,
                                  int64_t* simplified_a_dims,
                                  int64_t* simplified_b_dims) {
    const size_t num_max_dims = std::max(num_a_dims, num_b_dims);
    auto MakeGetDim = [num_max_dims](size_t num_dims, const int64_t* dims) {
        const int64_t num_padding_dims = num_max_dims - num_dims;
        return [num_padding_dims, dims](size_t index) { return index < num_padding_dims ? 1 : dims[index - num_padding_dims]; };
    };
    auto GetADim = MakeGetDim(num_a_dims, a_dims);
    auto GetBDim = MakeGetDim(num_b_dims, b_dims);
    *simplified_num_dims = 0;
    bool prev_broadcast_a = false;
    bool prev_broadcast_b = false;
    for (int64_t i = 0; i < num_max_dims; ++i) {
        const int64_t a_dim = GetADim(i);
        const int64_t b_dim = GetBDim(i);
        const int64_t broadcast_dim = std::max(a_dim, b_dim);
        const bool broadcast_a = (a_dim == 1);
        const bool broadcast_b = (b_dim == 1);
        if (broadcast_dim == 1) {
            continue;
        } else if (*simplified_num_dims != 0 && (prev_broadcast_a == broadcast_a && prev_broadcast_b == broadcast_b)) {
            simplified_a_dims[*simplified_num_dims - 1] *= a_dim;
            simplified_b_dims[*simplified_num_dims - 1] *= b_dim;
        } else {
            simplified_a_dims[*simplified_num_dims] = a_dim;
            simplified_b_dims[*simplified_num_dims] = b_dim;
            *simplified_num_dims += 1;
            prev_broadcast_a = broadcast_a;
            prev_broadcast_b = broadcast_b;
        }
    }
}

template <size_t num_dims, typename IndexType>
struct BroadcastMaskSoftmaxParams {
    NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
    NdIndexOffsetHelper<IndexType, num_dims> mask_index_helper;
    const int64_t* mask_dims{};
    int64_t row_size;
    float fill;
    float scale;
};

struct ElementwiseMaskSoftmaxParams {
    int64_t row_size;
    float fill;
    float scale;
};

template <typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct BroadcastScaleMaskLoad {
    BroadcastScaleMaskLoad(const SRC* src, const MASK* mask, BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
        : src(src), mask(mask), params(params) {
        for (int i = 0; i < num_dims; i++) {
            mask_dims[i] = params.mask_dims[i];
        }
    }
    template <int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) {
        Pack<SRC, N> pack;
        Pack<MASK, N> mask_pack;
        const IndexType offset = row * params.row_size + col;
        IndexType input_index[num_dims];
        IndexType mask_index[num_dims];
        params.src_index_helper.OffsetToNdIndex(offset, input_index);
        for (int dim = 0; dim < num_dims; ++dim) {
            if (mask_dims[dim] == 1) {
                mask_index[dim] = 0;
            } else {
                mask_index[dim] = input_index[dim];
            }
        }
        const IndexType mask_offset = params.mask_index_helper.NdIndexToOffset(mask_index);
        pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset / N);
        mask_pack.storage = *(reinterpret_cast<const PackType<MASK, N>*>(mask) + mask_offset / N);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (mask_pack.elem[i] == 0) {
                dst[i] = static_cast<DST>(params.fill);
            } else {
                dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(params.scale);
            }
        }
    }
    const SRC* src;
    const MASK* mask;
    int64_t mask_dims[num_dims];
    BroadcastMaskSoftmaxParams<num_dims, IndexType> params;
};

template <typename SRC, typename DST, typename MASK>
struct ElementwiseScaleMaskLoad {
    ElementwiseScaleMaskLoad(const SRC* src, const MASK* mask, ElementwiseMaskSoftmaxParams param) : src(src), mask(mask), param(param) {}
    template <int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) {
        Pack<SRC, N> pack;
        const int64_t offset = (row * param.row_size + col) / N;
        pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
        Pack<int8_t, N> mask_pack;
        mask_pack.storage = *(reinterpret_cast<const PackType<MASK, N>*>(mask) + offset);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (mask_pack.elem[i] == 0) {
                dst[i] = static_cast<DST>(param.fill);
            } else {
                dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(param.scale);
            }
        }
    }
    const SRC* src;
    const MASK* mask;
    ElementwiseMaskSoftmaxParams param;
};

template <typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct BroadcastScaleMaskStore {
    BroadcastScaleMaskStore(DST* dst, const MASK* mask, BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
        : dst(dst), mask(mask), params(params) {
        for (int i = 0; i < num_dims; ++i) {
            mask_dims[i] = params.mask_dims[i];
        }
    }
    template <int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> pack;
        Pack<MASK, N> mask_pack;
        const IndexType offset = row * params.row_size + col;
        IndexType input_index[num_dims];
        IndexType mask_index[num_dims];
        params.src_index_helper.OffsetToNdIndex(offset, input_index);
        for (int dim = 0; dim < num_dims; ++dim) {
            if (mask_dims[dim] == 1) {
                mask_index[dim] = 0;
            } else {
                mask_index[dim] = input_index[dim];
            }
        }
        const IndexType mask_offset = params.mask_index_helper.NdIndexToOffset(mask_index);
        mask_pack.storage = *(reinterpret_cast<const PackType<MASK, N>*>(mask) + mask_offset / N);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (mask_pack.elem[i] == 0) {
                pack.elem[i] = static_cast<DST>(params.fill);
            } else {
                pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(params.scale);
            }
        }
        *(reinterpret_cast<PackType<DST, N>*>(dst) + offset / N) = pack.storage;
    }
    DST* dst;
    const MASK* mask;
    int64_t mask_dims[num_dims];
    BroadcastMaskSoftmaxParams<num_dims, IndexType> params;
};

template <typename SRC, typename DST, typename MASK>
struct ElementwiseScaleMaskStore {
    ElementwiseScaleMaskStore(DST* dst, const MASK* mask, ElementwiseMaskSoftmaxParams params) : dst(dst), mask(mask), params(params) {}
    template <int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> pack;
        const int64_t offset = (row * params.row_size + col) / N;
        Pack<MASK, N> mask_pack;
        mask_pack.storage = *(reinterpret_cast<const PackType<MASK, N>*>(mask) + offset);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (mask_pack.elem[i] == 0) {
                pack.elem[i] = params.fill;
            } else {
                pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(params.scale);
            }
        }
        *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
    }
    DST* dst;
    const MASK* mask;
    ElementwiseMaskSoftmaxParams params;
};

template <typename SRC, typename DST>
struct MaskScaleLoad {
    MaskScaleLoad(const SRC* src, const bool* mask, int64_t row_size, SRC scale) : src(src), mask(mask), row_size(row_size), scale(scale) {}
    template <int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) const {
        Pack<SRC, N> pack;
        const int64_t offset = (row * row_size + col) / N;
        pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
        Pack<bool, N> mask_pack;
        mask_pack.storage = *(reinterpret_cast<const PackType<bool, N>*>(mask) + offset);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(mask_pack.elem[i]) * static_cast<DST>(scale);
        }
    }
    const SRC* src;
    const bool* mask;
    int64_t row_size;
    SRC scale;
};

template <typename SRC, typename DST>
struct DropoutStore {
    DropoutStore(DST* dst, DST* softmax_y, const bool* mask, int64_t row_size, DST scale)
        : dst(dst), softmax_y(softmax_y), mask(mask), row_size(row_size), scale(scale) {}
    template <int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> softmax_y_pack;
        Pack<DST, N> dst_pack;
        const int64_t offset = (row * row_size + col) / N;
        Pack<bool, N> mask_pack;
        mask_pack.storage = *(reinterpret_cast<const PackType<bool, N>*>(mask) + offset);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            softmax_y_pack.elem[i] = static_cast<DST>(src[i]);
            dst_pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(mask_pack.elem[i]) * static_cast<DST>(scale);
        }
        *(reinterpret_cast<PackType<DST, N>*>(softmax_y) + offset) = softmax_y_pack.storage;
        *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = dst_pack.storage;
    }
    DST* dst;
    DST* softmax_y;
    const bool* mask;
    int64_t row_size;
    DST scale;
};

template <typename T, typename ComputeType, typename MASK, size_t num_dims>
void LaunchBroadcastForwardKernel(cudaStream_t stream,
                                  const T* x,
                                  T* y,
                                  const MASK* mask,
                                  const int64_t elem_cnt,
                                  const int64_t rows,
                                  const int64_t cols,
                                  const float fill,
                                  const float scale,
                                  const int64_t* input_dims,
                                  const int64_t* mask_dims) {
    NdIndexOffsetHelper<int32_t, num_dims> input_index_helper(input_dims);
    NdIndexOffsetHelper<int32_t, num_dims> mask_index_helper(mask_dims);
    BroadcastMaskSoftmaxParams<num_dims, int32_t> params;
    params.src_index_helper = input_index_helper;
    params.mask_index_helper = mask_index_helper;
    params.mask_dims = mask_dims;
    params.row_size = cols;
    params.fill = fill;
    params.scale = scale;
    BroadcastScaleMaskLoad<T, ComputeType, MASK, num_dims, int32_t> load(x, mask, params);
    DirectStore<ComputeType, T> store(y, cols);
    (DispatchSoftmax<decltype(load), decltype(store), ComputeType>(stream, load, store, rows, cols));
    CUDA_CHECK();
}

template <typename T, typename ComputeType, typename MASK>
void LaunchElementwiseForwardKernel(cudaStream_t stream,
                                    const T* x,
                                    T* y,
                                    const MASK* mask,
                                    const int64_t rows,
                                    const int64_t cols,
                                    const float fill,
                                    const float scale) {
    ElementwiseMaskSoftmaxParams params;
    params.row_size = cols;
    params.fill = fill;
    params.scale = scale;
    ElementwiseScaleMaskLoad<T, ComputeType, MASK> load(x, mask, params);
    DirectStore<ComputeType, T> store(y, cols);
    (DispatchSoftmax<decltype(load), decltype(store), ComputeType>(stream, load, store, rows, cols));
    CUDA_CHECK();
}

int main() {
    const int batch_size = 4;
    const int num_heads = 8;
    const int seq_length = 64;
    const int N = batch_size * num_heads * seq_length * seq_length;

    float* x_host = (float*)malloc(N * sizeof(float));
    float* x_device;
    cudaMalloc((void**)&x_device, N * sizeof(float));
    for (int i = 0; i < N; i++)
        x_host[i] = 1.0;
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);

    float* y_host = (float*)malloc(N * sizeof(float));
    float* y_device;
    cudaMalloc((void**)&y_device, N * sizeof(float));

    bool* mask_host = (bool*)malloc(N * sizeof(bool));
    bool* mask_device;
    cudaMalloc((void**)&mask_device, N * sizeof(bool));
    for (int i = 0; i < N; i++)
        mask_host[i] = true;
    cudaMemcpy(mask_device, mask_host, N * sizeof(bool), cudaMemcpyHostToDevice);
    const float mask_fill_value = -10000.0;
    const float scale_value = 2.0;
    const int64_t cols = seq_length;
    const int64_t rows = N / seq_length;
    const size_t num_input_dims = 4;
    const int64_t input_dims[4] = {batch_size, num_heads, seq_length, seq_length};
    const size_t num_mask_dims = 4;
    const int64_t mask_dims[4] = {batch_size, num_heads, seq_length, seq_length};
    using ComputeType = typename DefaultComputeType<float>::type;
    size_t simplified_num_dims = 0;
    int64_t simplified_input_dims[4];
    int64_t simplified_mask_dims[4];

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    SimplifyBroadcastDims(num_input_dims, input_dims, num_mask_dims, mask_dims, &simplified_num_dims, simplified_input_dims, simplified_mask_dims);
    if (simplified_num_dims == 1) {
        LaunchElementwiseForwardKernel<float, ComputeType, bool>(stream, x_device, y_device, mask_device, rows, cols, mask_fill_value, scale_value);
    }
#define DEFINE_ONE_ELIF(dims)                                                                                                                 \
    else if (simplified_num_dims == dims) {                                                                                                   \
        LaunchBroadcastForwardKernel<float, ComputeType, bool, dims>(stream, x_device, y_device, mask_device, N, rows, cols, mask_fill_value, \
                                                                     scale_value, simplified_input_dims, simplified_mask_dims);               \
    }
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
#undef DEFINE_ONE_ELIF
    else {
        exit(-1);
    }
    cudaMemcpy(y_host, y_device, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32; i++) {
        printf("%.6f\n", y_host[i]);
    }
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(mask_device);
    free(x_host);
    free(y_host);
    free(mask_host);
    return 0;
}