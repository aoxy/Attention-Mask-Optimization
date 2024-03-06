all:
	nvcc -arch=sm_75 -o fused_scale_mask_softmax fused_scale_mask_softmax.cu
native:
	nvcc -arch=sm_75 -o fused_scale_mask_softmax_native fused_scale_mask_softmax_native.cu