
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Matrix.cuh"
#include "Timer.h"



int main()
{
    Timer* t=new Timer();
    Matrix* a = new Matrix(3, 3, new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
    a->Print();

    Matrix* b = new Matrix(3, 3, new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
    b->Print();
    
    t->Start();
    Matrix* sequential = *a * *b;
    printf("逐次　%f[ms]\n",t->Elapsed());
    sequential->Print();

    t->Start();
    a->ToDevice();
    b->ToDevice();
    Matrix* pallarel = Matrix::PallalelMulti(a,b);
    pallarel->ToHost();
    printf("並列　%f[ms]\n", t->Elapsed());
    pallarel->Print();

    return 0;
}
