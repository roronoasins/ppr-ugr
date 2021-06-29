/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace std;

int id, size, siguiente, anterior;
unsigned int NCIUDADES;
bool token_presente;

int main (int argc, char **argv) {
	int color_carga = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);


	switch (argc) {
		case 3:		NCIUDADES = atoi(argv[1]);
					break;
		default:	cerr << "La sintaxis es: bbseq <tama�o> <archivo>" << endl;
					exit(1);
					break;
	}

	int** tsp0 = reservarMatrizCuadrada(NCIUDADES);
	tNodo	nodo,         // nodo a explorar
			lnodo,        // hijo izquierdo
			rnodo,        // hijo derecho
			solucion;     // mejor solucion
	bool activo,			// condicion de fin
			 fin,
			 nueva_U;       // hay nuevo valor de c.s.
	int  U;             // valor de c.s.
	int iteraciones = 0;
	tPila pila;         // pila de nodos a explorar

	U = INFINITO;                  // inicializa cota superior
	InicNodo (&nodo);              // inicializa estructura nodo

  // el proceso 0 lee la matriz del fichero
  if(id==0)
	{
		LeerMatriz (argv[2], tsp0);    // lee matriz de fichero
		token_presente = true;				// el proceso 0 tiene inicialmente el testigo
	}

  // difusion de la matriz
  MPI_Bcast(&tsp0[0][0], NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

  // guardamos los procesos relevantes para reproducir el anillo en el equilibrado
  siguiente = (id+1)%size; anterior = (id-1+size)%size;
	// medida de tiempo
	MPI_Barrier(MPI_COMM_WORLD);
  double t=MPI_Wtime();
  if(id != 0)
  {
		token_presente = false;
    Equilibrar_Carga(pila, fin, solucion);
    if(!fin) pila.pop(nodo);
  }

	fin = Inconsistente(tsp0);
	while (!fin) {       // ciclo del Branch&Bound
		Ramifica (&nodo, &lnodo, &rnodo, tsp0); // para ramificar es necesario disponer de tsp0
		nueva_U = false;
		if (Solucion(&rnodo)) { // nodo raiz?
			if (rnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = rnodo.ci();     // actualizamos cota sup con cota inferior
				nueva_U = true;
				CopiaNodo (&rnodo, &solucion);  // si es hoja y mejor solucion, copiamos el nodo como nodo solucion
			}
		}
		else {                    //  no es un nodo solucion
			if (rnodo.ci() < U) {     //  cota inferior menor que cota superior
				if (!pila.push(rnodo)) { // si es prometedor se mete en la pila
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}
		if (Solucion(&lnodo)) {
			if (lnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = lnodo.ci();
				nueva_U = true;
				CopiaNodo (&lnodo,&solucion);
			}
		}
		else {                     // no es nodo solucion
			if (lnodo.ci() < U) {      // cota inferior menor que cota superior
				if (!pila.push(lnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}
		if (nueva_U) pila.acotar(U); // eliminar nodos cuya cota inf supera la superior(eliminamos nodos no prometedores)
    Equilibrar_Carga(pila, fin, solucion);
		if(!fin) pila.pop(nodo);
		iteraciones++;
	}
  t=MPI_Wtime()-t;
  MPI_Finalize();

	cout << "Numero de iteraciones = " << iteraciones << " en el proceso " << id <<endl << endl;
	if(id == 0)
	{
		cout<< "Tiempo gastado= "<<t<<endl;
		printf ("Solucion: \n");
		EscribeNodo(&solucion);
	}

	liberarMatriz(tsp0);
}
