#include "mpi.h"
#include <iostream>
using namespace std;

/* el proceso 0 difunde su identificador de proceso (0) al resto de procesos con identificadores pares, siguiendo
un anillo de procesos pares, y el proceso 1 hace lo mismo con los procesos impares*/

int main(int argc, char *argv[])
{
    int rank, dato, size;
    MPI_Status estado;

    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtenemos el valor de nuestro identificador
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtenemos el valor de nuestro identificador

    if(rank == 0 || rank == 1)
    {
      if(rank+2 <= size-1)
      {
        MPI_Send(&rank //referencia al vector de elementos a enviar
                ,1 // tamaño del vector a enviar
                ,MPI_INT // Tipo de dato que envias
                ,rank+2 // pid del proceso destino
                ,0 //etiqueta
                ,MPI_COMM_WORLD); //Comunicador por el que se manda
        cout<< "Soy el proceso "<<rank<<" y he enviado "<<rank<<endl;
      }
    }else{
      MPI_Recv(&dato // Referencia al vector donde se almacenara lo recibido
              ,1 // tamaño del vector a recibir
              ,MPI_INT // Tipo de dato que recibe
              ,rank-2 // pid del proceso origen de la que se recibe
              ,0 // etiqueta
              ,MPI_COMM_WORLD // Comunicador por el que se recibe
              ,&estado); // estructura informativa del estado
      cout<< "Soy el proceso "<<rank<<" y he recibido "<<dato<<endl;

      if(rank+2 <= size-1)
      {
        MPI_Send(&dato //referencia al vector de elementos a enviar
                ,1 // tamaño del vector a enviar
                ,MPI_INT // Tipo de dato que envias
                ,rank+2 // pid del proceso destino
                ,0 //etiqueta
                ,MPI_COMM_WORLD); //Comunicador por el que se manda
        cout<< "Soy el proceso "<<rank<<" y he enviado "<<rank<<endl;
      }
    }

    MPI_Finalize();
    return 0;
}
