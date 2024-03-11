#ifndef FUSED_GROUPED_TRIL_MASK_SOFTMAX_H_
#define FUSED_GROUPED_TRIL_MASK_SOFTMAX_H_

void fused_grouped_tril_mask_softmax(float* attn,
                                     float* result,
                                     const int64_t* seq_lens,
                                     const int64_t batch_size,
                                     const int64_t num_heads,
                                     const int64_t seq_length);

#endif  // FUSED_GROUPED_TRIL_MASK_SOFTMAX_H_
