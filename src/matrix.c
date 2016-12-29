#include "matrix.h"

Matrix* createMatrix(int rows, int cols, double** data){
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = data;
    return matrix;
}

Matrix* add(Matrix* A, Matrix* B){
    assert(A->rows == B->rows && A->cols == B->cols);
    double** data = (double**)malloc(sizeof(double*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (double*)malloc(sizeof(double) * A->cols);
    }
    Matrix* result = createMatrix(A->rows, A->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < A->cols; j++){
            data[i][j] = A->data[i][j] + B->data[i][j];
        }
    }
    return result;
}

Matrix* multiply(Matrix* A, Matrix* B){
    assert(A->cols == B->rows);
    double** data = (double**)malloc(sizeof(double*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (double*)malloc(sizeof(double) * B->cols);
    }
    Matrix* result = createMatrix(A->rows, B->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < B->cols; j++){
            double sum = 0;
            int k;
            for (k = 0; k < B->rows; k++){
                sum += A->data[i][k] * B->data[k][j];
            }
            data[i][j] = sum;
        }
    }
    return result;
}

Matrix* hadamard(Matrix* A, Matrix* B){
    assert(A->rows == B->rows);
    assert(A->cols == B->cols);
    double** data = (double**)malloc(sizeof(double*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (double*)malloc(sizeof(double) * A->cols);
    }
    Matrix* result = createMatrix(A->rows, A->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < A->cols; j++){
            data[i][j] = A->data[i][j] * B->data[i][j];
        }
    }
    return result;
}

Matrix* copy(Matrix* orig){
    double** data = (double**)malloc(sizeof(double*) * orig->rows);
    int i;
    for (i = 0; i < orig->rows; i++){
        data[i] = (double*)malloc(sizeof(double) * orig->cols);
    }
    int j;
    for (i = 0; i < orig->rows; i++){
        for (j = 0; j < orig->cols; j++){
            data[i][j] = orig->data[i][j];
        }
    }
    return createMatrix(orig->rows, orig->cols, data);
}

void destroyMatrix(Matrix* matrix){
    int i;
    for (i = 0; i < matrix->rows; i++){
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}