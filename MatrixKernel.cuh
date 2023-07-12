#ifndef _Kernel_
#define _Kernel_


#include "Matrix.cuh"

__global__ void Multiple(Matrix a, Matrix b, Matrix c);

Matrix* Lancher(Matrix* a, Matrix* b, Matrix* c);

#endif //kernel