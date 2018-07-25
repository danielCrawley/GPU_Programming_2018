#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>		// get time of day
#include <sys/times.h>		// get time of day
#include <sys/mman.h>		// mmap
#include <unistd.h>		// getpid
#include <cuda.h>
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

#define NUM_THREADS_PER_BLOCK_DIMENSION 	32
#define PRINT_TIME 				1
#define SM_ARR_LEN			2048
#define TOL						1e-1
#define OMEGA 1.8
#define GET_SECONDS_TICS 100	
#define IMUL(a, b) __mul24(a, b)

void initializeArray1D(float *arr, int len, int seed); // Initialize 2D array as concatenated sets of 1D arrays
void mmm_kij(float* a0, float* b0, float* c0, int length);         // MMM on host - fast ordering

double get_seconds() { 	/* routine to read time */
    struct tms rusage;
    times(&rusage);	/* UNIX utility: time in clock ticks */
    return (double)(rusage.tms_utime)/(double)(GET_SECONDS_TICS);
}


// CUDA kernel to perform MMM
__global__ void kernel_MMM (int arrLen, float* A, float* B, float* C) {

 for (int idx = blockDim.x*blockIdx.x + threadIdx.x; idx < arrLen; idx+=blockDim.x){
   for (int idy = blockDim.y*blockIdx.y + threadIdx.y; idy < arrLen; idy+=blockDim.y){
     int index = idx*arrLen + idy;
     if (index < arrLen*arrLen){            // check array bounds
     C[index] = 0.0;                      // initialize destination (not necessary?)
     float temp = 0.0;                    // per thread register as accumulator
     for (int k = 0; k < arrLen; k++) {   // per thread dot product for single output
		 temp += (A[idx*arrLen+k] * B[k*arrLen+idy]);
     }
     C[index] = temp;                     // save to output
   }
  }
 }
} // END CUDA kernel to perform MMM

int main(int argc, char **argv){
	int arrLen = 0;

	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;

	// Arrays on GPU global memoryc
	float *d_arrayA;
	float *d_arrayB;
	float *d_arrayC;

	// Arrays on the host memory
	float *h_arrayA;
	float *h_arrayB;
	float *h_arrayC_CPU;
	float *h_arrayC_GPU;
 
	int i, errCount = 0, zeroCount = 0;
	
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
  
  // Set block dimensions
  dim3 threadsPerBlock(NUM_THREADS_PER_BLOCK_DIMENSION, NUM_THREADS_PER_BLOCK_DIMENSION);
  dim3 NUM_BLOCKS(1,1);


	// Allocate GPU memory
	size_t allocSize = arrLen * arrLen * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayA, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayB, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayC, allocSize));
		
	// Allocate arrays on host memory
	h_arrayA                        = (float *) malloc(allocSize);
	h_arrayB                        = (float *) malloc(allocSize);
	h_arrayC_CPU                    = (float *) malloc(allocSize);
	h_arrayC_GPU                    = (float *) malloc(allocSize);
	
 
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducability
	initializeArray1D(h_arrayA, arrLen, 2453);
	initializeArray1D(h_arrayB, arrLen, 2453);
	printf("\t... done\n\n");
	
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
#endif

	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_arrayA, h_arrayA, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_arrayB, h_arrayB, allocSize, cudaMemcpyHostToDevice));
	
	// Launch the kernel
	cudaEventRecord(start, 0);
	kernel_MMM <<<NUM_BLOCKS, threadsPerBlock >>>(arrLen, d_arrayA, d_arrayB, d_arrayC);
  
	cudaEventRecord(stop,0);

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
   	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_arrayC_GPU, d_arrayC, allocSize, cudaMemcpyDeviceToHost));
 
 
#if PRINT_TIME
	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	
	// Compute the results on the host

  printf("Starting CPU Computation\n");
  struct timespec cpu_start, cpu_end; 
 
 clock_gettime(CLOCK_REALTIME, &cpu_start);	
  mmm_kij(h_arrayA, h_arrayB, h_arrayC_CPU, arrLen);
  clock_gettime(CLOCK_REALTIME, &cpu_end);
  float time_sec = cpu_end.tv_sec - cpu_start.tv_sec;
  float time_nsec = (cpu_end.tv_nsec - cpu_start.tv_nsec);  
  float time_msec = time_sec *1000 + time_nsec/1000000;
  printf("\n CPUTime = %f (msec)\n", time_msec);
  
	printf("Comparing Results\n");
	
	// Compare the results
	for(i = 0; i < arrLen*arrLen; i++) {
    if (abs((h_arrayC_CPU[i] - h_arrayC_GPU[i])/h_arrayC_GPU[i]) > TOL) {
			errCount++;
		}
		if (h_arrayC_GPU[i] == 0) {
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
	CUDA_SAFE_CALL(cudaFree(d_arrayA));
	CUDA_SAFE_CALL(cudaFree(d_arrayB));
	CUDA_SAFE_CALL(cudaFree(d_arrayC));
		   
	free(h_arrayC_CPU);
	free(h_arrayC_GPU);
	free(h_arrayA);
	free(h_arrayB);
		
	return 0;
}

void initializeArray1D(float *arr, int len, int seed) {
	int i;
	float randNum;
	srand(seed);

	for (i = 0; i < len*len; i++) {
		randNum = (float) rand();
		arr[i] = randNum;
   
	}
}


void mmm_kij(float* a0, float* b0, float* c0, int length)
{
  long int i, j, k;
  float r;
  
  for (k = 0; k < length; k++)
    for (i = 0; i < length; i++) {
      r = a0[i*length+k];
      for (j = 0; j < length; j++)
	c0[i*length+j] += r*b0[k*length+j];
    }
}
