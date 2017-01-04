#include "std_includes.h"

#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix_ {
    int rows;
    int cols;
    double** data;
} Matrix;

/*
    Generally, there are two types of functions here.
    
    Functions either take in arguments, and return the output,
    or they take in the neccesary arguments plus the matrix in 
    which the output is placed, and return nothing. The argument 
    ordering is generally: [operation arguments] [output]. 
*/

// creates a matrix given data
Matrix* createMatrix(int rows, int cols, double** data);

// uses memory of the original data to split matrix into submatrices
Matrix** createBatches(Matrix* allData, int numBatches);

// sets the values in $to equal to values in $from
void copyValuesInto(Matrix* from, Matrix* to);

// prints the entries of a matrix
void printMatrix(Matrix* input);

// sets each entry in matrix to 0
void zeroMatrix(Matrix* orig);

// returns transpose of matrix
Matrix* transpose(Matrix* orig);

// transposes matrix and places data into $origT
void transposeInto(Matrix* orig, Matrix* origT);

// collapses matrix into row vector of column averages
Matrix* columnAverages(Matrix* orig);

// adds two matrices and returns result
Matrix* add(Matrix* A, Matrix* b);

// adds $from to $to and places result in $to
void addTo(Matrix* from, Matrix* to);

// adds $B, a row vector, to each row of $A
Matrix* addToEachRow(Matrix* A, Matrix* B);

// multiplies every element of $orig by $C
void scalarMultiply(Matrix* orig, double c);

// multiplies $A and $B (ordering: AB) and returns product matrix
Matrix* multiply(Matrix* A, Matrix* B);

// multiplies $A and $B (ordering: AB) and places values into $into
void multiplyInto(Matrix* A, Matrix* B, Matrix* into);

// element-wise multiplcation
Matrix* hadamard(Matrix* A, Matrix* B);

// places values of hadamard product of $A and $B into $into
void hadamardInto(Matrix* A, Matrix* B, Matrix* into);

// returns a copy of input matrix
Matrix* copy(Matrix* orig);

// returns 1 if matrices are equal, 0 otherwise
int equals(Matrix* A, Matrix* B);

// frees a matrix and its data
void destroyMatrix(Matrix* matrix);


/*
    Begin functions.
*/

Matrix* createMatrix(int rows, int cols, double** data){
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = data;
    return matrix;
}

Matrix* createMatrixZeroes(int rows, int cols){
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    double** data = (double**)calloc(rows, sizeof(double*));
    int i;
    for (i = 0; i < rows; i++){
        data[i] = (double*)calloc(cols, sizeof(double));
    }
    matrix->data = data;
    return matrix;
}

Matrix** createBatches(Matrix* allData, int numBatches){
    Matrix** batches = (Matrix**)malloc(sizeof(Matrix*) * numBatches);
    int remainder = allData->rows % numBatches;
    int i;
    int curRow = 0;
    for (i = 0; i < numBatches; i++){
        int batchSize = allData->rows / numBatches;
        if (remainder-- > 0){
            batchSize++;
        }
        batches[i] = createMatrix(batchSize, allData->cols, allData->data + curRow);
        curRow += batchSize;
    }
    return batches;
}

void copyValuesInto(Matrix* from, Matrix* to){
    assert(from->rows == to->rows && from->cols == to->cols);
    int i;
    for (i = 0; i < from->rows; i++){
        memcpy(to->data[i], from->data[i], sizeof(double) * from->cols);
    }
}

void printMatrix(Matrix* input){
    int i, j;
    for (i = 0; i < input->rows; i++){
        printf("\n");
        for (j = 0; j < input->cols; j++){
            printf("%.2f ", input->data[i][j]);
        }
    }
    printf("\n");
}

void zeroMatrix(Matrix* orig){
    int i;
    for (i = 0; i < orig->rows; i++){
        memset(orig->data[i], 0, orig->cols * sizeof(double));
    }
}

Matrix* transpose(Matrix* orig){
    double** data = (double**)malloc(sizeof(double*) * orig->cols);
    int k;
    for (k = 0; k < orig->cols; k++){
        data[k] = (double*)malloc(sizeof(double) * orig->rows);
    }
    Matrix* transpose = createMatrix(orig->cols, orig->rows, data);
    int i, j;
    for (i = 0; i < orig->rows; i++){
        for (j = 0; j < orig->cols; j++){
            transpose->data[j][i] = orig->data[i][j];
        }
    }
    return transpose;
}

void transposeInto(Matrix* orig, Matrix* origT){
    assert(orig->rows == origT->cols && orig->cols == origT->rows);
    int i, j;
    for (i = 0; i < orig->rows; i++){
        for (j = 0; j < orig->cols; j++){
            origT->data[j][i] = orig->data[i][j];
        }
    }
}

Matrix* columnAverages(Matrix* orig){
    double** data = (double**)malloc(sizeof(double*));
    data[0] = (double*)malloc(sizeof(double) * orig->cols);
    int i, j;
    for (i = 0; i < orig->cols; i++){
        double colSum = 0;
        for (j = 0; j < orig->rows; j++){
            colSum += orig->data[j][i];
        }
        data[0][i] = colSum / orig->rows;
    }
    return createMatrix(1, orig->cols, data);
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

void addTo(Matrix* from, Matrix* to){
    assert(from->rows == to->rows && from->cols == to->cols);
    int i, j;
    for (i = 0; i < from->rows; i++){
        for (j = 0; j < from->cols; j++){
            to->data[i][j] += from->data[i][j];
        }
    }
}

// add B to each row of A
Matrix* addToEachRow(Matrix* A, Matrix* B){
    assert(A->cols == B->cols && B->rows == 1);
    double** data = (double**)malloc(sizeof(double*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (double*)malloc(sizeof(double) * A->cols);
    }
    Matrix* result = createMatrix(A->rows, A->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < A->cols; j++){
            data[i][j] = A->data[i][j] + B->data[0][j];
        }
    }
    return result;
}

void scalarMultiply(Matrix* orig, double c){
    int i, j;
    for (i = 0; i < orig->rows; i++){
        for (j = 0; j < orig->cols; j++){
            orig->data[i][j] *= c;
        }
    }
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

void multiplyInto(Matrix* A, Matrix* B, Matrix* into){
    assert(A->cols == B->rows);
    assert(A->rows == into->rows && B->cols == into->cols);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < B->cols; j++){
            double sum = 0;
            int k;
            for (k = 0; k < B->rows; k++){
                sum += A->data[i][k] * B->data[k][j];
            }
            into->data[i][j] = sum;
        }
    }
}

Matrix* hadamard(Matrix* A, Matrix* B){
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
            data[i][j] = A->data[i][j] * B->data[i][j];
        }
    }
    return result;
}

void hadamardInto(Matrix* A, Matrix* B, Matrix* into){
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(A->rows == into->rows && A->cols == into->cols);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < A->cols; j++){
            into->data[i][j] = A->data[i][j] * B->data[i][j];
        }
    }
}

Matrix* copy(Matrix* orig){
    double** data = (double**)malloc(sizeof(double*) * orig->rows);
    int i;
    for (i = 0; i < orig->rows; i++){
        data[i] = (double*)malloc(sizeof(double) * orig->cols);
        memcpy(data[i], orig->data[i], sizeof(double) * orig->cols);
    }
    return createMatrix(orig->rows, orig->cols, data);
}

int equals(Matrix* A, Matrix* B){
    if (A->rows != B->rows){
        return 0;
    }
    if (A->cols != B->cols){
        return 0;
    }
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < A->cols; j++){
            if (A->data[i][j] != B->data[i][j]){
                return 0;
            }
        }
    }
    return 1;
}

void destroyMatrix(Matrix* matrix){
    int i;
    for (i = 0; i < matrix->rows; i++){
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

#endif