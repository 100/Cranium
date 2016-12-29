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

Matrix* addToEachRow(Matrix* A, Matrix* B);

Matrix* multiply(Matrix* A, Matrix* B);

// element-wise multiplcation
Matrix* hadamard(Matrix* A, Matrix* B);

// returns a copy of input matrix
Matrix* copy(Matrix* orig);

void destroyMatrix(Matrix* matrix);