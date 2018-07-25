#include <cstdio>
#include <cstdlib>
#include <math.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define NUM_THREADS_PER_BLOCK 	256
#define NUM_BLOCKS 				1
#define PRINT_TIME 				1
#define SM_ARR_LEN				50000
#define TOL						1e-6
#define ITERS          1000

#define IMUL(a, b) __mul24(a, b)

void initializeArray1D(float *arr, int len, int seed);

__global__ void kernel_add (int arrLen, float* x, float* y, float* result) {
	const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = IMUL(blockDim.x, gridDim.x);
	
	int i,j;
	for(i = tid; i < arrLen; i += threadN) {
     float temp = 0;
     float x_local = x[i];
     float y_local = y[i];
     for (j = 0 ; j < ITERS; j+= 1){                          // vary ITERS to vary AI
		    temp += (1e-6 * x_local ) + (1e-7 * y_local) + 0.25;  // use accumulator to prevent loop simplification at compile time, use local variables to hold memory intensity constant
     }
     result[i] = temp;                                        
  }
}

int main(int argc, char **argv){
	int arrLen = 0;
		
	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	
	// Arrays on GPU global memoryc
	float *d_x;
	float *d_y;
	float *d_result;

	// Arrays on the host memory
	float *h_x;
	float *h_y;
	float *h_result;
	float *h_result_gold;
	
	int i, j , errCount = 0, zeroCount = 0;
	
	if (argc > 1) {
		arrLen  = atoi(argv[1]);
	}
	else {
		arrLen = SM_ARR_LEN;
	}

	printf("Length of the array = %d\n", arrLen);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));
int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
	// Allocate GPU memory
	size_t allocSize = arrLen * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));
		
	// Allocate arrays on host memory
	h_x                        = (float *) malloc(allocSize);
	h_y                        = (float *) malloc(allocSize);
	h_result                   = (float *) malloc(allocSize);
	h_result_gold              = (float *) malloc(allocSize);
	
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducability
	initializeArray1D(h_x, arrLen, 2453);
	initializeArray1D(h_y, arrLen, 1467);
	printf("\t... done\n\n");
	
	
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	
#endif
	
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, allocSize, cudaMemcpyHostToDevice));
	

	// Record event on the default stream
	cudaEventRecord(start, 0);
  
	// Launch the kernel
	kernel_add<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(arrLen, d_x, d_y, d_result);

	cudaEventRecord(stop,0);
	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));
	
#if PRINT_TIME
	// Stop and destroy the timer
	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	
	// Compute the results on the host
	for(i = 0; i < arrLen; i++) {
    h_result_gold[i] = 0;
    for (j = 0 ; j < ITERS; j++){
		  h_result_gold[i] += (1e-6 * h_x[i]) + (1e-7 * h_y[i]) + 0.25;
	  }
  }
	
	// Compare the results
	for(i = 0; i < arrLen; i++) {
		if (abs(h_result_gold[i] - h_result[i]) > TOL) {
			errCount++;
		}
		if (h_result[i] == 0) {
			zeroCount++;
		}
	}
	
	/*
	for(i = 0; i < 50; i++) {
		printf("%d:\t%.8f\t%.8f\n", i, h_result_gold[i], h_result[i]);
	}
	*/
	
	if (errCount > 0) {
		printf("\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
	}
	else if (zeroCount > 0){
		printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	else {
		printf("\nTEST PASSED: All results matched\n");
	}
	
	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
	CUDA_SAFE_CALL(cudaFree(d_result));
		   
	free(h_x);
	free(h_y);
	free(h_result);
		
	return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
	int i;
	float randNum;
	srand(seed);

	for (i = 0; i < len; i++) {
		randNum = (float) rand();
		arr[i] = randNum;
	}
}
