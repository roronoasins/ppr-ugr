#include <math.h>
#include <cstdlib> // atoi
#include <iostream>
#include "mpi.h"
using namespace std;

/*
solución del cálculo paralelo del número π, los subintervalos de trabajo deben ser distribuidos por bloques en lugar de cíclicamente entre los procesos.
Modificarlo también la solución para que la aproximación a π se obtenga en todos los procesos.
*/

int main(int argc, char *argv[])
{
  int rank, dato, size;
  MPI_Status estado;

  MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el valor de nuestro identificador
  int n;
  if(rank == 0)
  {
    cout<<"introduce la precision del calculo (n > 0): ";
    cin>>n;
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  double PI25DT = 3.141592653589793238462643;
  double h = 1.0 / (double) n;
  double sum = 0.0;

  int istart = (n/size)*rank+1, iend = (n/size)*(rank+1);
  cout << "istart: " << istart << " iend:  " << iend << " para " << rank << endl;
  for (int i = istart; i <= iend; i++) {
  	double x = h * ((double)i - 0.5);
  	sum += (4.0 / (1.0 + x*x));
  }

  double pi_parcial = sum * h;
  double pi_reduce;
  MPI_Allreduce(&pi_parcial, &pi_reduce, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  cout << "El valor aproximado de PI es: " << pi_reduce << ", con un error de " << fabs(pi_reduce -PI25DT) << " para el proceso: " << rank << endl;

  MPI_Finalize();
  return 0;

}
