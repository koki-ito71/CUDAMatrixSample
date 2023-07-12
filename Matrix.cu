
#include "MatrixKernel.cuh"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Matrix::Matrix(){
	Set(0, 0);
}


Matrix::Matrix(int r, int c):Matrix(r,c,0.0){

}

Matrix::Matrix(int r, int c, float value){
	Set(r,c);

	int length = r * c;
	elements = new float[length];
	for (int i = 0; i < length; i++) {
		elements[i] = value;
	}
}
Matrix::Matrix(int r, int c, float* values) :Matrix(r, c) {
	Set(r, c);
	elements = values;
}
void Matrix::Set(int r, int c) {
	rowSize = r;
	columnSize = c;
	place = Place::Host;
}

Matrix::~Matrix() {
	if (place == Place::Host) {
		delete[] elements;
	}
	else {
		cudaFree(elements);
	}
}

Place Matrix::GetPlace() {
	return place;
}

Matrix* Matrix::CreateDevice(int r, int c) {
	Matrix* m = new Matrix();
	m->rowSize = r;
	m->columnSize = c;
	m->place = Place::Device;
	int size = r * c * sizeof(float);
	void* device;
	cudaMalloc(&device, size);
	m->elements = (float*)device;
	return m;
}

int Matrix::PosToIndex(int r, int c) {
	return r * columnSize +c; 
}

void Matrix::SetValue(int r, int c, float value) {
	elements[PosToIndex(r, c)] = value;
}

float Matrix::GetValue(int r, int c) {
	return elements[PosToIndex(r, c)];
}

void Matrix::ToDevice() {
	if (place == Place::Device)return;

	int size = rowSize * columnSize * sizeof(float);
	void* device;
	cudaMalloc(&device, size);
	cudaMemcpy(device, elements, size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	delete[] elements;
	elements = (float*)device;
	place = Place::Device;
}

void Matrix::ToHost() {
	if (place == Place::Host)return;

	int size = rowSize * columnSize * sizeof(float);
	float* host = new float[rowSize * columnSize];
	cudaMemcpy(host, elements, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(elements);
	elements = host;
	place = Place::Host;
}

Matrix* Matrix::operator +(const Matrix& m) {
	if (rowSize != m.rowSize || columnSize != m.columnSize) {
		throw std::exception("大きさが異なる");
	}
	Matrix* result = new Matrix(rowSize, columnSize);
	for (int i = 0; i < rowSize * columnSize; i++)
	{
		result->elements[i] = elements[i] + m.elements[i];
	}
	return result;
}

Matrix* Matrix::operator -(const Matrix& m) {
	if (rowSize != m.rowSize || columnSize != m.columnSize) {
		throw std::exception("大きさが異なる");
	}
	Matrix* result = new Matrix(rowSize, columnSize);
	for (int i = 0; i < rowSize * columnSize; i++)
	{
		result->elements[i] = elements[i] - m.elements[i];
	}
	return result;
}
Matrix* Matrix::operator *(Matrix& m) {
	if (columnSize!= m.rowSize) {
		throw std::exception("大きさが異なる");
	}
	Matrix* result = new Matrix(rowSize, m.columnSize);

	for (int i = 0; i < rowSize; i++) {
		for (int j = 0; j < m.columnSize; j++) {
			int temp = 0;
			for (int k = 0; k < columnSize; k++) {
				temp += GetValue(i, k) * m.GetValue(k, j);
			}
			result->SetValue(i, j, temp);
		}
	}
	return result;
}


Matrix* Matrix::PallalelMulti(Matrix* a, Matrix* b) {
	if (a->GetPlace() == Place::Host || b->GetPlace() == Place::Host) {
		throw std::exception("データがデバイス上にない");
	}
	Matrix* c = Matrix::CreateDevice(a->rowSize, b->columnSize);
	Lancher(a, b, c);
	return c;
}

void Matrix::Print() {
	if (place == Place::Device) {
		printf("woarning Device memory\n");
		return;
	}
	printf("(%d * %d)\n",rowSize,columnSize);
	printf("[\n");
	for (int i = 0; i < rowSize; i++) {
		printf("  [");

		for (int j = 0; j < columnSize; j++) {
			printf("%f", GetValue(i, j));
			if (j > 8) {
				printf("  ....");
				break;
			}
			if (j < columnSize - 1)printf(" , ");
		}
		printf("]\n");
		if (i > 8) {
			printf("\n  ........\n");
			break;
		}
	}
	printf("]\n");
}