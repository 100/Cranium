// code for matrix operations
#include <assert.h>
#include <stdlib.h>

typedef struct Matrix_ {
    int rows;
    int cols;
    double** data;
} Matrix;

Matrix* createMatrix(int rows, int cols, double** data);

Matrix* add(Matrix* A, Matrix* b);

Matrix* multiply(Matrix* A, Matrix* B);

void destroyMatrix(Matrix* matrix);