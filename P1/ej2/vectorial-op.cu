#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

int index(int i){return  i+2;}
// Blocksize
#define  BLOCKSIZE 256
#define blockSizeRed 32


//*************************************************
// Swap two pointers to float
// ************************************************
void swap_pointers(float * *a, float * *b)
     {float * tmp=*a;*a=*b;*b=tmp;}


//*************************************************
// GLOBAL MEMORY  VERSION OF THE FD UPDATE
// ************************************************
__global__ void vectorial_operation(float * d_a,
                           float * d_b,
                           int n)
  {
    int i=threadIdx.x+blockDim.x*blockIdx.x+2;

    // Inner point update
    if (i<n+2)
    {
      d_b[i-2]= (d_a[i-2]*d_a[i-2]+2.0*(d_a[i-1]*d_a[i-1]) + d_a[i]*d_a[i]
                   -3.0*(d_a[i+1]*d_a[i+1]) + 5.0*(d_a[i+2]*d_a[i+2]))/24.0;
    }
  }



//*************************************************
// TILING VERSION  (USES SHARED MEMORY) OF THE FD UPDATE
// ************************************************
__global__ void vectorial_operation_shared_mem(float * input, float * output, int n)
  {
    int li=threadIdx.x+2;   //local index in shared memory vector
    int gi=blockDim.x*blockIdx.x+threadIdx.x+2; // global memory index
    int lend=BLOCKSIZE+2;
    __shared__ float sdata[BLOCKSIZE + 4];  //shared mem. vector
    float result;

   // Load Tile in shared memory
    if (gi<n+2) sdata[li]=input[gi];

   if (threadIdx.x <= 1) // First two threads (in the current block)
          sdata[threadIdx.x]=input[gi-2];

   if (threadIdx.x == BLOCKSIZE-2 || threadIdx.x == BLOCKSIZE-1)  // Last two threads
	  if (gi==n+2)  // Last Block
	      sdata[threadIdx.x]=input[n+2];
	  else if (gi==n+3) // Last Block
        sdata[threadIdx.x]=input[n+3];
    else
	      sdata[lend]=input[gi+1];

  __syncthreads();

   if (gi<n+2)
    {
      result=(sdata[li-2]*sdata[li-2]+2.0*(sdata[li-1]*sdata[li-1]) + sdata[li]*sdata[li]
                -3.0*(sdata[li+1]*sdata[li+1]) + 5.0*(sdata[li+2]*sdata[li+2]))/24.0;
      output[gi-2]=result;
    }

  }

  //*************************************************
  // MAXIMUM REDUCTION WITHIN BLOCKS
  // ************************************************
  __global__ void reduceMax(float * M_in, float * M_out, const int N) {
    extern __shared__ float sdata[];

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


//******************************
//**** MAIN FUNCTION ***********

int main(int argc, char* argv[]){

//******************************
  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err=cudaGetDevice(&devID);
  if (err!=cudaSuccess) {cout<<"ERRORRR"<<endl;}
  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
    devID, props.name, props.major, props.minor);

  int n;
  cout<<"Introduce number of points (1000-200000)"<<endl;
  cin>>n;

  //Mesh Definition    blockDim.x*blockIdx.x
  float * A = new float[n+4];
  float * B = new float[n];
  float * cpu = new float[n+4];

  // A(-2) to A(n+1)
  for(int i=0;i<=n-1;i++)
     A[index(i)]= (float) (1  -(index(i)%100)*0.001);
  // Impose Boundary Conditions
  A[index(-2)] = A[index(-1)] = A[index(n)] = A[index(n+1)] = 0;

  //**************************
  // GPU phase
  //**************************
  int size=(n+4)*sizeof(float);

  // Allocation in device mem. for d_phi
  float * d_A=NULL;
  err=cudaMalloc((void **) &d_A, size);
  if (err!=cudaSuccess) {cout<<"ALLOCATION ERROR d_A"<<endl;}
  // Allocation in device mem. for d_B
  float * d_B=NULL;
  err=cudaMalloc((void **) &d_B, n*sizeof(float));
  if (err!=cudaSuccess) {cout<<"ALLOCATION ERROR d_B"<<endl;}

  // Take initial time
  double  t1=clock();

  // Copy phi values to device memory
  err=cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);

  if (err!=cudaSuccess) {cout<<"GPU COPY ERROR"<<endl;}

  int blocksPerGrid =(int) ceil((float)(n+4)/BLOCKSIZE);

  vectorial_operation<<<blocksPerGrid, BLOCKSIZE>>> (d_A, d_B, n);
  cudaDeviceSynchronize();
  double Tgpu=clock();
  Tgpu=(Tgpu-t1)/CLOCKS_PER_SEC;

  cout<< "GPU Time= "<<Tgpu<<endl<<endl;

  //**************************
  // Reduction phase
  //**************************

  dim3 threadsPerBlockRed(blockSizeRed, 1);
  dim3 numBlocksRed( ceil ((float)(n)/threadsPerBlockRed.x), 1);
  int smemSize = blockSizeRed*sizeof(float);
  float * a_out = NULL;
  err = cudaMalloc((void **) &a_out, numBlocksRed.x*sizeof(float));
  if (err != cudaSuccess) {
    cout << "ERROR RESERVA a_out" << endl;
  }

  reduceMax<<<numBlocksRed, threadsPerBlockRed, smemSize>>>(d_B, a_out, n);

  float *a_reduction = new float[numBlocksRed.x];
  cudaMemcpy(B, d_B, n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(a_reduction, a_out, numBlocksRed.x*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  float max_gpu = 0;
  //cout << "contenido de la reduccion" << endl;
  for (int i=0; i<numBlocksRed.x; i++)
  {
  //  cout << a_reduction[i] << " ";
    max_gpu = max(max_gpu,a_reduction[i]);
  }
  cout <<endl;

  float max_B = 0;
  //cout << "contenido de B" << endl;
  for (int i=0; i<n; i++)
  {
  //  cout << B[i] << " ";
    max_B = max(max_B,B[i]);
  }
  cout <<endl;

  cout << "max B: " << max_B << endl;
  cout << "max gpu: " << max_gpu << endl;

  cudaFree(d_A);cudaFree(d_A);cudaFree(a_out);
  free(A);free(B);free(a_reduction);

  for(int i=0;i<=n;i++)
    {
      float phi_i   = A[index(i)];
      float phi_ip1 = A[index(i+1)];
      float phi_im1 = A[index(i-1)];
      float phi_ip2 = A[index(i+2)];
      float phi_im2 = A[index(i-2)];

      cpu[index(i)]= (phi_im2*phi_im2+2.0*(phi_im1*phi_im1) + phi_i*phi_i
                   -3.0*(phi_ip1*phi_ip1) + 5.0*(phi_ip2*phi_ip2))/24.0;
    }

  int passed=1;
  int i=0;
  while (passed && i<n)
  {
    double diff=fabs((double)B[i]-(double)cpu[index(i)]);
    if (diff>1.0e-5)
    {
      passed=0;
        cout <<"DIFF= "<<diff<<endl;}
    i++;
  }

  if (passed)
   cout<<"PASSED TEST !!!"<<endl;
  else
   cout<<"ERROR IN TEST !!!"<<endl;

  return 0;
}
