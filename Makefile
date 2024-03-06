all:
	nvcc -arch=sm_75 -o fused_scale_mask_softmax fused_scale_mask_softmax.cu