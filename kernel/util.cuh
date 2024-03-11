#ifndef ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
#define ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_

namespace oneflow {

#define OF_CUDA_CHECK(condition)                                                               \
  for (cudaError_t _of_cuda_check_status = (condition); _of_cuda_check_status != cudaSuccess;) \
  std::cout << "Check failed: " #condition " : " << cudaGetErrorString(_of_cuda_check_status)  \
             << " (" << _of_cuda_check_status << ") " << std::endl

#define CHECK_GT(lhs, rhs) \
    if ((lhs) <= (rhs))    \
    std::cout << "Check failed: " #lhs " <= " #rhs << std::endl

#define CHECK(condition) \
    if (!(condition))    \
    std::cout << "Check failed: " #condition << std::endl

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_