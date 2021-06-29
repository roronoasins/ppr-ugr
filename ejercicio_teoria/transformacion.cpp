#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

using namespace std;

int index(int i){return  i+1;}

void swap_pointers(float * *a, float * *b)
     {float * tmp=*a;*a=*b;*b=tmp;}

 int mod(int a, int b) {
   int c = a % b;
   return (c < 0) ? c + b : c;
 }

int main(int argc, char* argv[])
{
  int n, rank, size, which;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc < 2) {
      if(rank == 0)
        cout << "No se ha especificado numero de elementos, por defecto será: " <<  size * 100;
      n = size * 100;
  } else {
    n = atoi(argv[1]);
  }

  float * V               = new float[n+2];
  float * V_secuencial    = new float[n+2];
  float * V_new           = new float[n+2];
  float * V_final         = new float[n+2];

  if (rank==0)
  {
    srand( (unsigned)time( NULL ) );
    for(int i=0; i < n; ++i)
    {
      V[index(i)] = (float) rand()/RAND_MAX;
      V_secuencial[index(i)] = V[index(i)];
    }V[index(-1)] = V[index(n)] = 0;V_secuencial[index(-1)] = 0; V_secuencial[index(n)] = 0;V_new[index(-1)] = 0; V_new[index(n)] = 0;

    // secuencial
    for(int k=0;k<100;k++)
    {
      //V_secuencial[index(-1)] = V_secuencial[index(n)];
      for(int i=0;i<n;i++)
      {
        float V_i   = V_secuencial[index(i)];
        float V_ip1 = V_secuencial[index(i+1)];
        float V_im1 = V_secuencial[index(i-1)];

        V_new[index(i)] = (V_im1 - V_i + V_ip1) / 2;
      }
      swap_pointers (&V_secuencial,&V_new);
      ++which;
    }// fin secuencial
  }

  // calculo mpi
  MPI_Status status;
  int tam_bloq = (n+2)/size;
  float * Bloque = new float[tam_bloq];
  MPI_Scatter(V, tam_bloq, MPI_FLOAT, Bloque, tam_bloq, MPI_FLOAT, 0, MPI_COMM_WORLD);

  float izq, der, tmp;
  int dest_im1 = mod(rank-1, size);
  int dest_ip1 = mod(rank+1, size);
  for(int k=0;k<100;k++)
  {
    //cout << "Envio " << Bloque[0] << " a " << dest_im1 << " desde " << rank << endl;
    MPI_Send(&Bloque[0], 1, MPI_FLOAT, dest_im1, 0, MPI_COMM_WORLD);
    //cout << "Recibiendo de " << dest_ip1 << " desde " << rank << endl;
    MPI_Recv(&der, 1, MPI_FLOAT, dest_ip1, 0, MPI_COMM_WORLD, &status);
    //cout << "Recibido de " << dest_ip1 << " desde " << rank << endl;

    //cout << "Envio " << Bloque[tam_bloq-1] << " a " << dest_ip1 << " desde " << rank << endl;
    MPI_Send(&Bloque[tam_bloq-1], 1, MPI_FLOAT, dest_ip1, 1, MPI_COMM_WORLD);
    //cout << "Recibiendo de " << dest_im1 << " desde " << rank << endl;
    MPI_Recv(&izq, 1, MPI_FLOAT, dest_im1, 1, MPI_COMM_WORLD, &status);
    //cout << "Recibido de " << dest_im1 << " desde " << rank << endl;

    int j = ( rank == 0 ) ? 1 : 0;
    for(j; j <= tam_bloq-2; ++j)
    {
      tmp = Bloque[j];
      Bloque[j] = (izq - Bloque[j] + Bloque[j+1])/2;
      izq = tmp;
    }
    if (rank!=size-1)
      Bloque[tam_bloq-1] = (izq - Bloque[tam_bloq-1] + der)/2;
  }// fin calculo mpi

  //for(int i=0; i<tam_bloq;++i)
    //cout << "desde " << rank << " con i=" << i << " valor: Bloque=" << Bloque[i] << endl;

  MPI_Gather(Bloque, tam_bloq, MPI_FLOAT, V_final, tam_bloq, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank==0)
  {
    int passed=1;
    int i=0;
    int a=0, b=0;
    float* v = mod(which,2) == 0 ? V_new : V_secuencial;
    while (passed && i<n+1)
    {
      double diff=fabs((double)v[index(i)]-(double)V_final[index(i)]);
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

     /*if(mod(which,2) == 0)
     {
       for(int i=0; i<n+2;++i)
         cout << "i: " << i << " V-secuencial: "<< V_secuencial[i] << endl;
     }else{
       for(int i=0; i<n+2;++i)
         cout << "i: " << i << " V-secuencial: "<< V_new[i] << endl;
     }

     for(int i=0; i<n+2;++i)
       cout << "i: " << i << " V-final: "<< V_final[i] << endl;*/
  }



  MPI_Finalize();
  return 0;
}
