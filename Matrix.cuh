#ifndef _Matrix_
#define _Matrix_


using namespace std;
#include <string>
#include <stdio.h>
#include <exception> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

enum class Place{Device,Host};

struct Matrix {

private:
	Matrix();
	void Set(int r, int c);
	Place place;

public:
	int rowSize;
	int columnSize;

	float* elements;


	Matrix(int r, int c);

	Matrix(int r, int c, float value);

	Matrix(int r, int c, float* values);

	static Matrix* CreateDevice(int r, int c);

	~Matrix();

	Place GetPlace();

	int Matrix::PosToIndex(int r, int c);

	void Matrix::SetValue(int r, int c, float value);

	float Matrix::GetValue(int r, int c);

	void Matrix::ToDevice();

	void Matrix::ToHost();

	Matrix* operator +(const Matrix& m);

	Matrix* operator -(const Matrix& m);
	Matrix* operator *(Matrix& m);

	static Matrix* PallalelMulti(Matrix* a, Matrix* b);

	void Print();
};






#endif //model