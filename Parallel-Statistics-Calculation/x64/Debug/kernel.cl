__kernel void reduce_min(__global const int* in, __global int* out, __local int* cache) {
    //Get local variable data
	int lid = get_local_id(0);
    int id = get_global_id(0);
    int N = get_local_size(0);

    //Store valus of global memory into local memory
    cache[lid] = in[id];

    //Wait for copying to complete
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (!(lid % (i * 2)) && ((lid + i) < N)){
            if (cache[lid] > cache[lid + i])
                cache[lid] = cache[lid+i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Store cache in output array
    if (!lid)
        atomic_min(&out[0], cache[lid]);
}


__kernel void reduce_max(__global const int* in, __global int* out, __local int* cache) {
    //Get local variable data
    int lid = get_local_id(0);
	int id = get_global_id(0);
    int N = get_local_size(0);

    //Store valus of global memory into local memory
    cache[lid] = in[id];

    //Wait for copying to complete
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (!(lid % (i * 2)) && ((lid + i) < N)){
            if (cache[lid] < cache[lid + i])
                cache[lid] = cache[lid+i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Store cache in output array
    if (!lid)
        atomic_max(&out[0], cache[lid]);
}

__kernel void reduce_sum(__global const int* in, __global int* out, __local int* cache) {
	//Get local variable data
	int lid = get_local_id(0);
	int id = get_global_id(0);
	int N = get_local_size(0);

	//Store valus of global memory into local memory
	cache[lid] = in[id];

  //Wait for copying to complete
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
				cache[lid] += cache[lid+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Store cache in output array
	if (!lid)
		atomic_add(&out[0], cache[lid]);
}

__kernel void hist(__global const int* A, __global int* H,  int nr_bins, int initial) {			
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id+initial];//take value as a bin index

	if(bin_index < nr_bins)
		atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}