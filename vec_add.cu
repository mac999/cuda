

void vecAdd(const float *h_A, const float *h_B, float *h_C, int numElements)
{
    // Allocate the device input vectors A, B, C
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, numElements * sizeof(float)));
 
    // Copy the host input vector A and B in host memory 
    // to the device input vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice));
 
    // Allocate CUDA events for estimating
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
 
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
 
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
 
    CUDA_CHECK(cudaGetLastError());
 
    // Copy the device result vector in device memory
    // to the host result vector in host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    CUDA_CHECK(cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify that the result vector is correct (sampling)
    printf("Verifying vector addition...\n");
    for (int idx = 0; idx < numElements; idx++) {
        //printf("[INDEX %d] %f + %f = %f\n", idx, h_A[idx], h_B[idx], h_C[idx]);
        if (fabs(h_A[idx] + h_B[idx] - h_C[idx]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d\n", idx);
            exit(EXIT_FAILURE);
        }
    }
    printf(".....\n");
    printf("Test PASSED\n");
 
    // Compute and Print the performance
    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));
    double flopsPerVecAdd = static_cast<double>(numElements);
    double gigaFlops = (flopsPerVecAdd * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size = %.0f Ops, "
           "WorkgroupSize= %u threads/block\n",
           gigaFlops, msecTotal, flopsPerVecAdd, threadsPerBlock);
    
    // Free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}