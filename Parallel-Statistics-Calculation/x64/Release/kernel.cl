__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

__kernel void equal(__global const float* A, __global float* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

__kernel void reduce_min(__global const double* A, __global double* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	
	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N))
		{
			if(B[id] > B[id + i])
			{
				B[id] = B[id + i];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void reduce_max(__global const double* A, __global double* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	
	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N))
		{
			if(B[id] < B[id + i])
			{
				B[id] = B[id + i];
			}
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}