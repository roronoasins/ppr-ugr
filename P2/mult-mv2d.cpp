#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include "mpi.h"

using namespace std;

int main(int argc, char * argv[]) {

  int p, rank;
  long **A, // Matriz a multiplicar
          *x, // Vector que vamos a multiplicar
          *y, // Vector donde almacenamos el resultado
          *comprueba; // Guarda el resultado final (calculado secuencialmente), su valor
                      // debe ser igual al de 'y'

  double tInicio, // Tiempo en el que comienza la ejecucion
          tFin; // Tiempo en el que acaba la ejecucion

  if(argc != 2)
  {
    cout << "Error en la ejecución. Paramétros esperados: <número de filas/cols de la matriz nxn>" << endl;
    exit(-1);
  }
  int n = atoi(argv[1]);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int raiz_p = sqrt(p), tam = n/raiz_p;
  int row_color  = rank/raiz_p,
      col_color  = rank%raiz_p;
  MPI_Comm diag_comm;
  int diag_color=(row_color == col_color) ? 0 : MPI_UNDEFINED, diag_p, diag_rank;
  MPI_Comm row_comm, col_comm;
  MPI_Comm_split(MPI_COMM_WORLD, row_color, rank, &row_comm);
  MPI_Comm_split(MPI_COMM_WORLD, col_color, rank, &col_comm);
  MPI_Comm_split(MPI_COMM_WORLD, diag_color, rank, &diag_comm);

  int row_rank, col_rank, row_p, col_p;
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_size(row_comm, &row_p);
  MPI_Comm_rank(col_comm, &col_rank);
  MPI_Comm_size(col_comm, &col_p);
  MPI_Comm_rank(diag_comm, &diag_rank);
  MPI_Comm_size(diag_comm, &diag_p);

  MPI_Datatype MPI_BLOQUE;

  /*if(rank == 0 && n%p != 0)
  {
    cout << "Error en la ejecución. Se espera un número de filas/cols múltiplo de " << p << "." << endl;
    exit(-1);
  }*/
  A = new long *[n]; // Reservamos tantas filas como procesos haya
  x = new long [n]; // El vector sera del mismo tamanio que el numero
  // de procesadores
  long *buf_envio = new long[n*n];

  // Solo el proceso 0 ejecuta el siguiente bloque
  if (rank == 0) {
    A[0] = new long [n * n];
    for (unsigned int i = 1; i < n; i++) {
        A[i] = A[i - 1] + n;
    }
    // Reservamos especio para el resultado
    y = new long [n];

    // Rellenamos 'A' y 'x' con valores aleatorios
    srand(time(0));
    //cout << "La matriz y el vector generados son " << endl;
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            //if (j == 0) cout << "[";
            A[i][j] = rand() % 1000;
            //cout << A[i][j];
            //if (j == n - 1) cout << "]";
            //else cout << "  ";
        }
        x[i] = rand() % 100;
        //cout << "\t  [" << x[i] << "]" << endl;
    }
    //cout << "\n";

    // Reservamos espacio para la comprobacion
    comprueba = new long [n];
    // Lo calculamos de forma secuencial
    for (unsigned int i = 0; i < n; i++) {
        comprueba[i] = 0;
        for (unsigned int j = 0; j < n; j++) {
            comprueba[i] += A[i][j] * x[j];
        }
    }

    MPI_Type_vector(tam, tam, n, MPI_LONG, &MPI_BLOQUE);
    MPI_Type_commit(&MPI_BLOQUE);
    int comienzo, posicion=0;
    for(int i=0; i<p; ++i)
    {
      row_p = i/raiz_p;
      col_p = i%raiz_p;
      MPI_Pack(&A[row_p*tam][col_p*tam], 1, MPI_BLOQUE, buf_envio, sizeof(long)*n*n, &posicion, MPI_COMM_WORLD);
    }
    MPI_Type_free(&MPI_BLOQUE);
  } // Termina el trozo de codigo que ejecuta solo 0

  long *buf_recep = new long[tam*tam];
  MPI_Scatter(buf_envio, sizeof(long)*tam*tam, MPI_PACKED, buf_recep, tam*tam, MPI_LONG, 0, MPI_COMM_WORLD);
  //for(int i=0; i<tam*tam; ++i)
    //cout << "buf_recep: " << buf_recep[i] << " desde " << rank << endl;
  long *xlocal = new long [tam];
  MPI_Scatter(x, tam, MPI_LONG, xlocal, tam, MPI_LONG, 0, diag_comm);
  MPI_Bcast(xlocal, tam, MPI_LONG, row_rank, col_comm);
  //for(int i=0; i<tam; ++i)
    //cout << "xlocal: " << xlocal[i] << " desde " << rank << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  // Inicio de medicion de tiempo
  tInicio = MPI_Wtime();

  long *ylocal = new long[tam];
  for(int i=0; i<tam; ++i)
  {
    ylocal[i] = 0;
    for(int j=0; j<tam; ++j)
    {
      ylocal[i] += buf_recep[j+tam*i] * xlocal[j];
    }
  }

  // Otra barrera para asegurar que todas ejecuten el siguiente trozo de codigo lo
  // mas proximamente posible
  MPI_Barrier(MPI_COMM_WORLD);
  // fin de medicion de tiempo
  tFin = MPI_Wtime();

  long *y_row_reduce = new long[tam];

  MPI_Reduce(ylocal, y_row_reduce, tam, MPI_LONG, MPI_SUM, col_rank, row_comm);
  long *y_gather = new long[n];

  MPI_Gather(y_row_reduce, tam, MPI_LONG, y_gather, tam, MPI_LONG, 0, diag_comm);
  //for(int i=0; i<tam; ++i)
    //cout << "y_row_reduce: " << y_row_reduce[i] << " desde " << rank << endl;
  MPI_Finalize();

  if (rank == 0) {

    unsigned int errores = 0;

    cout << "El resultado obtenido y el esperado son:" << endl;
    for (unsigned int i = 0; i < n; i++) {
        cout << "\t" << y_gather[i] << "\t|\t" << comprueba[i] << endl;
        if (comprueba[i] != y_gather[i])
            errores++;
    }

    delete [] y;
    delete [] comprueba;
    delete [] A[0];

    if (errores) {
        cout << "Hubo " << errores << " errores." << endl;
    } else {
        cout << "No hubo errores" << endl;
        cout << "El tiempo tardado ha sido " << tFin - tInicio << " segundos." << endl;
    }

  }

  delete [] x;
  delete [] A;
  delete [] xlocal;
  delete [] ylocal;
  delete [] y_gather;

}
