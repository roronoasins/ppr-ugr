all: mult-mv mult-mv2d secuencial

secuencial: secuencial.cpp
	mpicxx -o secuencial secuencial.cpp

mult-mv: mult-mv.cpp
	mpicxx -o mult-mv mult-mv.cpp

mult-mv2d: mult-mv2d.cpp
	mpicxx -o mult-mv2d mult-mv2d.cpp

clean:
	rm mult-mv mult-mv2d secuencial
