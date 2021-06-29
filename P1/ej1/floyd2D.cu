#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define blockSize2D 32
#define blockSizeRed 1024

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************
__global__ void floyd2D_kernel(int * M, const int nverts, const int k) {
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < nverts && j < nverts) {
		int ij = i*nverts+j;
		int Mij = M[ij];
    if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
    	Mij = (Mij > Mikj) ? Mikj : Mij;
    	M[ij] = Mij;
		}
  }
}

__global__ void reduceMax(int * M_in, int * M_out, const int N) {
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? M_in[i] : 0.0f);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){
	  if (tid < s)
		if(sdata[tid] < sdata[tid+s])
                    sdata[tid] = sdata[tid+s];
	  __syncthreads();
	}
	if (tid == 0)
           M_out[blockIdx.x] = sdata[0];
}

int main (int argc, char *argv[]) {

	if (argc != 2) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}


  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if(err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}


	cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]);// Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

	// GPU phase
	double  t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	dim3 threadsPerBlock (blockSize2D, blockSize2D);
	dim3 numBlocks( ceil ((float)(nverts)/threadsPerBlock.x), ceil ((float)(nverts)/threadsPerBlock.y) );

	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	  floyd2D_kernel<<<numBlocks,threadsPerBlock >>>(d_In_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	
	double Tgpu = cpuSecond()-t1;

	cout << "Tiempo gastado GPU= " << Tgpu << endl << endl;

	// CPU phase
	t1 = cpuSecond();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}

  double t2 = cpuSecond() - t1;
  cout << "Tiempo gastado CPU= " << t2 << endl << endl;
  cout << "Ganancia= " << t2 / Tgpu << endl;


  for(int i = 0; i < nverts; i++)
    for(int j = 0;j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
         cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;

	int * d_in_M_reduction = NULL;
	err = cudaMalloc((void **) &d_in_M_reduction, size);
	err = cudaMemcpy(d_in_M_reduction, c_Out_M, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	dim3 threadsPerBlockRed(blockSizeRed, 1);
	dim3 numBlocksRed( ceil ((float)(nverts2)/threadsPerBlockRed.x), 1);
	int smemSize = blockSizeRed*sizeof(int);
	int * d_out_M_reduction = NULL;
	err = cudaMalloc((void **) &d_out_M_reduction, numBlocksRed.x*sizeof(int));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	reduceMax<<<numBlocksRed, threadsPerBlockRed, smemSize>>>(d_in_M_reduction ,d_out_M_reduction, nverts2);

	int redsize = ceil ((float)(nverts2)/threadsPerBlockRed.x);
	int *out_reduction = new int[numBlocksRed.x];
	cudaMemcpy(out_reduction, d_out_M_reduction, numBlocksRed.x*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int max_cpu = 0;
	for (int i=0; i<nverts2;i++){
	  max_cpu=max(max_cpu, c_Out_M[i]);
  }

	int max_gpu = 0;
	for (int i=0; i<numBlocksRed.x; i++)
	{
	  max_gpu =max(max_gpu,out_reduction[i]); //printf(" d_out_reduction[%d]=%i\n",i,d_out_reduction[i]);
	}
	printf("%i\n", max_gpu);
	printf("%i\n", max_cpu);

	int passed=1;
	int i=0;
	while (passed && i<nverts2)
	{
		double diff=fabs((double)A[i]-(double)c_Out_M[i]);
		if (diff>1.0e-5)
		{ passed=0;
				cout <<"DIFF= "<<diff<<endl;}
		i++;
	}

	if (passed)
	 cout<<"PASSED TEST !!!"<<endl;
	else
	 cout<<"ERROR IN TEST !!!"<<endl;

}
