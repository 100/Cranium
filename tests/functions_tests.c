#include "../src/functions.c"
#include <stdio.h>

int main(){
    // test sigmoid
    double k;
    for (k = -10.0; k < 30.0; k += 2){
        assert(sigmoidFunc(k) >= 0 && sigmoidFunc(k) <= 1);
    }

    // test softmax
    double** row = (double**)malloc(sizeof(double*));
    row[0] = (double*)malloc(sizeof(double) * 10);
    int j;
    for (j = 0; j < 10; j++){
        row[0][j] = exp(j/2);
        if (j % 2 == 0){
            row[0][j] *= -1;
        }
    }
    Matrix* rowMatrix = createMatrix(1, 10, row);
    softmax(rowMatrix);
    double sum = 0;
    for (j = 0; j < 10; j++){
        assert(rowMatrix->data[0][j] >= 0 && rowMatrix->data[0][j] <= 1);
        sum += rowMatrix->data[0][j];
    }
    assert(sum >= .99 && sum <= 1.01);

    // test cross-entropy loss
    double** A_data = (double**)malloc(sizeof(double*) * 3);
    int i;
    for (i = 0; i < 3; i++){
        A_data[i] = (double*)malloc(sizeof(double) * 3);
        for (j = 0; j < 3; j++){
            A_data[i][j] = i + j;
        }
    }

    double** B_data = (double**)malloc(sizeof(double*) * 3);
    for (i = 0; i < 3; i++){
        B_data[i] = (double*)malloc(sizeof(double) * 3);
        for (j = 0; j < 3; j++){
            B_data[i][j] = i + j;
        }
    }

    Matrix* predict = createMatrix(3, 3, A_data);
    Matrix* actual = createMatrix(3, 3, B_data);
    assert(crossEntropyLoss(predict, actual) == 0);

    destroyMatrix(predict);
    destroyMatrix(actual);

    return 0;
}