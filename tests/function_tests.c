#include "../src/std_includes.h"
#include "../src/matrix.h"
#include "../src/function.h"

int main(){
    // test sigmoid
    float k;
    for (k = -10.0; k < 30.0; k += 2){
        assert(sigmoidFunc(k) >= 0 && sigmoidFunc(k) <= 1);
    }

    // test relu
    for (k = -10.0; k < 30.0; k += 2){
        assert(reluFunc(k) == MAX(0, k));
    }

    // test softmax
    float* row = (float*)malloc(sizeof(float) * 10);
    int j;
    for (j = 0; j < 10; j++){
        row[j] = expf(j/2);
        if (j % 2 == 0){
            row[j] *= -1;
        }
    }
    Matrix* rowMatrix = createMatrix(1, 10, row);
    softmax(rowMatrix);
    float sum = 0;
    for (j = 0; j < 10; j++){
        assert(rowMatrix->data[j] >= 0 && rowMatrix->data[j] <= 1);
        sum += rowMatrix->data[j];
    }
    assert(sum >= .99 && sum <= 1.01);

    destroyMatrix(rowMatrix);

    return 0;
}