#include "../src/std_includes.h"
#include "../src/matrix.h"
#include "../src/function.h"

int main(){
    // test sigmoid
    double k;
    for (k = -10.0; k < 30.0; k += 2){
        assert(sigmoidFunc(k) >= 0 && sigmoidFunc(k) <= 1);
    }

    // test relu
    for (k = -10.0; k < 30.0; k += 2){
        assert(reluFunc(k) == MAX(0, k));
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

    destroyMatrix(rowMatrix);

    return 0;
}