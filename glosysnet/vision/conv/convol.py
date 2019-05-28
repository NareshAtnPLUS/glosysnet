import numpy as np

def conv2D(image,conv_filter,padding=0,stride=1):
	sub_array_dim = ((image.shape[0] + 2*padding - conv_filter.shape[0])//stride)+1

	sub_shape = (sub_array_dim,sub_array_dim)

	view_shape = tuple(np.subtract(image.shape, sub_shape) + 1) + sub_shape
	strides = image.strides + image.strides

	sub_matrices = np.lib.stride_tricks.as_strided(image,view_shape,strides)

	m = np.einsum('ij,ijkl->kl',conv_filter,sub_matrices)

	return m 
def pooling(convoled_mat,pooling_size):
	view_shape = tuple(np.subtract(convoled_mat.shape, pooling_size) + 1) + pooling_size
	strides = convoled_mat.strides + convoled_mat.strides
	sub_matrices = np.lib.stride_tricks.as_strided(convoled_mat,view_shape,strides)
	max_pool = np.zeros((pooling_size[0]+1,pooling_size[0]+1))
	for i in range(len(sub_matrices)):
		for j in range(len(sub_matrices[i])):
			max_pool[i][j] = ((sub_matrices[i][j]).max())
	return max_pool