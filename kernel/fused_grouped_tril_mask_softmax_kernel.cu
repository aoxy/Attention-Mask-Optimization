#include "softmax.cuh"
#include "fused_softmax.cuh"
#include "util.cuh"

namespace oneflow {

template<typename SRC, typename DST>
struct GroupedTrilMaskLoad {
  GroupedTrilMaskLoad(const SRC* src, const int64_t* seq_lens, const int64_t row_size, const int64_t batch_rows, const SRC fill)
      : src(src),
        seq_lens(seq_lens),
        row_size(row_size),
        batch_rows(batch_rows),
        fill(fill) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    const int64_t seq_len = seq_lens[row / batch_rows];
    const int64_t inside_row = row % row_size;
    const bool invalid = (inside_row >= seq_len || col >= seq_len);
    const bool need_load = (!invalid && inside_row >= col);
    cuda::softmax::Pack<SRC, N> pack;
    if (need_load) {
      const int64_t offset = (row * row_size + col) / N;
      pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src) + offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (invalid || inside_row < col + i) {
        dst[i] = static_cast<DST>(fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]);
      }
    }
  }
  const SRC* src;
  const int64_t* seq_lens;
  const int64_t row_size;
  const int64_t batch_rows;
  const SRC fill;
};

template<typename SRC, typename DST>
struct GroupedTrilMaskStore {
  GroupedTrilMaskStore(DST* dst, const int64_t* seq_lens, const int64_t row_size, const int64_t batch_rows, const DST fill)
      : dst(dst),
        seq_lens(seq_lens),
        row_size(row_size),
        batch_rows(batch_rows),
        fill(fill) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    const int64_t seq_len = seq_lens[row / batch_rows];
    const int64_t inside_row = row % row_size;
    const bool invalid = (inside_row >= seq_len || col >= seq_len);
    cuda::softmax::Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (invalid || inside_row < col + i) {
        pack.elem[i] = fill;
      } else {
        pack.elem[i] = static_cast<DST>(src[i]);
      }
    }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  const int64_t* seq_lens;
  const int64_t row_size;
  const int64_t batch_rows;
  const DST fill;
};

template<typename T, typename ComputeType>
void LaunchGroupedTrilMaskSoftmaxKernel(cudaStream_t stream, const T* x, T* y, const int64_t* seq_lens,
                                    const int64_t rows, const int64_t cols, const int64_t batch_rows,
                                    const float fill) {
  GroupedTrilMaskLoad<T, ComputeType> load(x, seq_lens, cols, batch_rows, fill);
  GroupedTrilMaskStore<ComputeType, T> store(y, seq_lens, cols, batch_rows, static_cast<T>(0.0));
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      stream, load, store, rows, cols)));
}

}  // namespace oneflow

void fused_grouped_tril_mask_softmax(float* attn,
                                     float* result,
                                     const int64_t* seq_lens,
                                     const int64_t batch_size,
                                     const int64_t num_heads,
                                     const int64_t seq_length) {
    const int64_t batch_rows = num_heads * seq_length;
    const int64_t rows = batch_size * batch_rows;
    const int64_t cols = seq_length;
    const float fill = -1e10f;
    using ComputeType = typename oneflow::cuda::softmax::DefaultComputeType<float>::type;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    oneflow::LaunchGroupedTrilMaskSoftmaxKernel<float, ComputeType>(stream, attn, result, seq_lens, rows, cols, batch_rows, fill);
}

                