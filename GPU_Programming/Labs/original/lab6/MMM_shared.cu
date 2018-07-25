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


#define NUM_THREADS_PER_BLOCK_DIMENSION 	16
#define PRINT_TIME 				1
#define SM_ARR_LEN			2048
#define TOL						1e-1
#define OMEGA 1.8
#define GET_SECONDS_TICS 100	
#define IMUL(a, b) __mul24(a, b)

void initializeArray1D(float *arr, int len, int seed); // Initialize 2D array as concatenated sets of 1D arrays
void mmm_kij(float* a0, float* b0, float* c0);

double get_seconds() { 	/* routine to read time */
    struct tms rusage;
    times(&rusage);	/* UNIX utility: time in clock ticks */
    return (double)(rusage.tms_utime)/(double)(GET_SECONDS_TICS);
}


__global__ void kernel_MMM (int arrLen, float* A, float* B, float* C) { 

 __shared__ float tempA[NUM_THREADS_PER_BLOCK_DIMENSION][NUM_THREADS_PER_BLOCK_DIMENSION];
 __shared__ float tempB[NUM_THREADS_PER_BLOCK_DIMENSION][NUM_THREADS_PER_BLOCK_DIMENSION];

int tx = threadIdx.x;
int ty = threadIdx.y;
int bx = blockIdx.x;
int by = blockIdx.y;
int Col = bx*blockDim.x + tx;
int Row = by*blockDim.y + ty;
float Pvalue = 0.0;

for (int m =0 ; m < arrLen/NUM_THREADS_PER_BLOCK_DIMENSION; m++){
  tempA[ty][tx] = A[Row*arrLen + (m*NUM_THREADS_PER_BLOCK_DIMENSION + tx)];
  tempB[ty][tx] = B[Col + (m*NUM_THREADS_PER_BLOCK_DIMENSION + ty)*arrLen];
  __syncthreads();
  
  for (int k = 0; k < NUM_THREADS_PER_BLOCK_DIMENSION; k++)
    Pvalue += tempA[ty][k] * tempB[k][tx];

  __syncthreads();
}
C[Row*arrLen+Col] = Pvalue;
}




int main(int argc, char **argv){
	int arrLen = 0;

	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	double sec;
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
  
  // Set block dimensions
  dim3 threadsPerBlock(NUM_THREADS_PER_BLOCK_DIMENSION, NUM_THREADS_PER_BLOCK_DIMENSION);
  dim3 NUM_BLOCKS(arrLen/NUM_THREADS_PER_BLOCK_DIMENSION,arrLen/NUM_THREADS_PER_BLOCK_DIMENSION);


	// Allocate GPU memory
	size_t allocSize = arrLen * arrLen * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayA, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayB, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_arrayC, allocSize));
		
	// Allocate arrays on host memory
	h_arrayA                        = (float *) malloc(allocSize);
	h_arrayB                        = (float *) malloc(allocSize);
	h_arrayC_CPU                        = (float *) malloc(allocSize);
	h_arrayC_GPU                        = (float *) malloc(allocSize);
	
 
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
	cudaEventRecord(start, 0);
#endif

	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_arrayA, h_arrayA, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_arrayB, h_arrayB, allocSize, cudaMemcpyHostToDevice));
	

	// Launch the kernel
	kernel_MMM <<<NUM_BLOCKS, threadsPerBlock >>>(arrLen, d_arrayA, d_arrayB, d_arrayC);
  
	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
   	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_arrayC_GPU, d_arrayC, allocSize, cudaMemcpyDeviceToHost));
 
 
#if PRINT_TIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	
 
 
	// Compute the results on the host
  printf("Starting CPU Computation\n");
  sec = get_seconds();		
  long int  j, k;
  float r;
  int length = arrLen;
  for (k = 0; k < length; k++)
    for (i = 0; i < length; i++) {
      r = h_arrayA[i*length+k];
      for (j = 0; j < length; j++)
	      h_arrayC_CPU[i*length+j] += r*h_arrayB[k*length+j];
    }
  sec = (get_seconds() - sec);	
  printf("\n CPUTime = %f (msec)\n", sec*1000.0);

   
   
  // Compare the results
	printf("Comparing Results\n");
	for(i = 0; i < arrLen*arrLen; i++) {
		if (abs((h_arrayC_CPU[i] - h_arrayC_GPU[i])/h_arrayC_GPU[i]) > TOL) {
			errCount++;
		}
		if (h_arrayC_GPU[i] == 0) {
			zeroCount++;
		}
	}
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
