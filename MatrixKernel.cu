#include "MatrixKernel.cuh"

__global__ void Multiple(float *a, float* b, float* c,int a_row)
{
    int row =  blockIdx.x;
    int col = blockIdx.y;

    float x = 0.0f;
    //printf("(%d,%d)  %f = %f * %f\n", row, col, x, a[col * gridDim.x], b[row]);

    for (int k = 0; k <a_row; k++) {
        x += a[k + col * a_row] * b[row + k * gridDim.x];
    }
    c[row + col * gridDim.x] = x;
}
__global__ void Multiple2(float *a, float* b, float* c,int a_col)
{
    int row =  blockIdx.x;
    int col = threadIdx.x;

    float x = 0.0f;
    //printf("(%d,%d)  %f = %f * %f\n", row, col, x, a[col * blockDim.x], b[row]);

    for (int k = 0; k <a_col; k++) {
        x += a[row * a_col + k] * b[k * blockDim.x + col];
    }
    c[row * blockDim.x +col] = x;
}
__global__ void Multiple3(float *a, float* b, float* c,int a_row)
{
    int row = blockIdx.x; 
    int col = threadIdx.x;

    float x = 0.0f;
    //printf("(%d,%d)  %f = %f * %f\n", row, col, x, a[col * blockDim.x], b[row]);

    for (int k = 0; k <a_row; k++) {
        x += a[k + col * a_row] * b[row + k * blockDim.x];
    }
    c[row + col * blockDim.x] = x;
}

Matrix* Lancher(Matrix* a, Matrix* b, Matrix* c) {
    //dim3 grid(b->row, a->column), block(1, 1);
    dim3 grid( a->rowSize,1), block(b->columnSize, 1);
    Multiple2 << <grid,block>> > (a->elements, b->elements, c->elements,a->columnSize);
    return c;
}