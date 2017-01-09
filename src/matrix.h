#include "std_includes.h"

#ifndef MATRIX_H
#define MATRIX_H

// represents a matrix of data
typedef struct Matrix_ {
    size_t rows;
    size_t cols;
    float** data;
} Matrix;

/*
    Generally, there are two types of functions here.
    
    Functions either take in arguments, and return the output,
    or they take in the neccesary arguments plus the matrix in 
    which the output is placed, and return nothing. The argument 
    ordering is generally: [operation arguments] [output]. 
*/

// creates a matrix given data
static Matrix* createMatrix(size_t rows, size_t cols, float** data);

// uses memory of the original data to split matrix into submatrices
static Matrix** createBatches(Matrix* allData, int numBatches);

// sets the values in $to equal to values in $from
static void copyValuesInto(Matrix* from, Matrix* to);

// prints the entries of a matrix
static void printMatrix(Matrix* input);

// sets each entry in matrix to 0
static void zeroMatrix(Matrix* orig);

// returns transpose of matrix
static Matrix* transpose(Matrix* orig);

// transposes matrix and places data into $origT
static void transposeInto(Matrix* orig, Matrix* origT);

// collapses matrix into row vector of column averages
static Matrix* columnAverages(Matrix* orig);

// adds two matrices and returns result
static Matrix* add(Matrix* A, Matrix* b);

// adds $from to $to and places result in $to
static void addTo(Matrix* from, Matrix* to);

// adds $B, a row vector, to each row of $A
static Matrix* addToEachRow(Matrix* A, Matrix* B);

// multiplies every element of $orig by $C
static void scalarMultiply(Matrix* orig, float c);

// multiplies $A and $B (ordering: AB) and returns product matrix
static Matrix* multiply(Matrix* A, Matrix* B);

// multiplies $A and $B (ordering: AB) and places values into $into
static void multiplyInto(Matrix* A, Matrix* B, Matrix* into);

// element-wise multiplcation
static Matrix* hadamard(Matrix* A, Matrix* B);

// places values of hadamard product of $A and $B into $into
static void hadamardInto(Matrix* A, Matrix* B, Matrix* into);

// returns a shallow copy of input matrix
static Matrix* copy(Matrix* orig);

// returns 1 if matrices are equal, 0 otherwise
static int equals(Matrix* A, Matrix* B);

// shuffle two matrices, maintaining alignment between their rows
static void shuffleTogether(Matrix* A, Matrix* B);

// frees a matrix and its data
static void destroyMatrix(Matrix* matrix);


/*
    Begin functions.
*/

Matrix* createMatrix(size_t rows, size_t cols, float** data){
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
    float** data = (float**)calloc(rows, sizeof(float*));
    int i;
    for (i = 0; i < rows; i++){
        data[i] = (float*)calloc(cols, sizeof(float));
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
        size_t batchSize = allData->rows / numBatches;
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
        memcpy(to->data[i], from->data[i], sizeof(float) * from->cols);
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
        memset(orig->data[i], 0, orig->cols * sizeof(float));
    }
}

Matrix* transpose(Matrix* orig){
    float** data = (float**)malloc(sizeof(float*) * orig->cols);
    int k;
    for (k = 0; k < orig->cols; k++){
        data[k] = (float*)malloc(sizeof(float) * orig->rows);
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
    float** data = (float**)malloc(sizeof(float*));
    data[0] = (float*)malloc(sizeof(float) * orig->cols);
    int i, j;
    for (i = 0; i < orig->cols; i++){
        float colSum = 0;
        for (j = 0; j < orig->rows; j++){
            colSum += orig->data[j][i];
        }
        data[0][i] = colSum / orig->rows;
    }
    return createMatrix(1, orig->cols, data);
}

Matrix* add(Matrix* A, Matrix* B){
    assert(A->rows == B->rows && A->cols == B->cols);
    float** data = (float**)malloc(sizeof(float*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (float*)malloc(sizeof(float) * A->cols);
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
    float** data = (float**)malloc(sizeof(float*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (float*)malloc(sizeof(float) * A->cols);
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

void scalarMultiply(Matrix* orig, float c){
    int i, j;
    for (i = 0; i < orig->rows; i++){
        for (j = 0; j < orig->cols; j++){
            orig->data[i][j] *= c;
        }
    }
}

Matrix* multiply(Matrix* A, Matrix* B){
    assert(A->cols == B->rows);
    float** data = (float**)malloc(sizeof(float*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (float*)malloc(sizeof(float) * B->cols);
    }
    Matrix* result = createMatrix(A->rows, B->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < B->cols; j++){
            float sum = 0;
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
            float sum = 0;
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
    float** data = (float**)malloc(sizeof(float*) * A->rows);
    int k;
    for (k = 0; k < A->rows; k++){
        data[k] = (float*)malloc(sizeof(float) * A->cols);
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
    float** data = (float**)malloc(sizeof(float*) * orig->rows);
    int i;
    for (i = 0; i < orig->rows; i++){
        data[i] = (float*)malloc(sizeof(float) * orig->cols);
        memcpy(data[i], orig->data[i], sizeof(float) * orig->cols);
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

void shuffleTogether(Matrix* A, Matrix* B){
    assert(A->rows == B->rows);
    int i;
    for (i = 0; i < A->rows - 1; i++){
        size_t j = i + rand() / (RAND_MAX / (A->rows - i) + 1);
        float* tmpA = A->data[j];
        A->data[j] = A->data[i];
        A->data[i] = tmpA;
        float* tmpB = B->data[j];
        B->data[j] = B->data[i];
        B->data[i] = tmpB;
    }
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