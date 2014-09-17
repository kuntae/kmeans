
CXX=mpic++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq opencl pthread openmp opencl_multi mpi mpi_opencl

.PHONY: all seq opencl clean


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -g -o $@ ${LDFLAGS}


opencl: kmeans_opencl

kmeans_opencl: kmeans_opencl.o kmeans_main.o
	${CXX} $^ -g -o $@ ${LDFLAGS} -lOpenCL

opencl_multi: kmeans_opencl_multi

kmeans_opencl_multi: kmeans_opencl_multi.o kmeans_main.o
	${CXX} $^ -g -o $@ ${LDFLAGS} -lOpenCL

pthread: kmeans_pthread

kmeans_pthread: kmeans_pthread.o kmeans_main.o
	${CXX} $^ -g -o $@ ${LDFLAGS} -lpthread

openmp: kmeans_openmp

kmeans_openmp: kmeans_openmp.o kmeans_main.o
	${CXX} $^ -g -o $@ ${LDFLAGS} -fopenmp

kmeans_openmp.o: kmeans_openmp.cpp
	${CXX} $^ -g -c ${CXXFLAGS} -fopenmp

mpi: kmeans_mpi

kmeans_mpi: kmeans_mpi.o kmeans_main.o
	${CXX} $^ -g -o $@ ${CXXFLAGS} -lOpenCL

clean:
	rm -f kmeans_opencl_multi kmeans_opencl_multi.o kmeans kmeans_seq kmeans_opencl kmeans_pthread kmeans_openmp kmeans_main.o kmeans_seq.o kmeans_opencl.o kmeans_pthread.o kmeans_openmp.o

