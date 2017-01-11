#include "../src/std_includes.h"
#include "../src/matrix.h"

int main(){
    float* A_data = (float*)malloc(sizeof(float*) * 3 * 3);
    int i, j;
    for (i = 0; i < 3; i++){
        for (j = 0; j < 3; j++){
            A_data[i * 3 + j] = i + j;
        }
    }

    float* B_data = (float*)malloc(sizeof(float) * 3 * 4);
    for (i = 0; i < 3; i++){
        for (j = 0; j < 4; j++){
            B_data[i * 4 + j] = i + j;
        }
    }

    // test creation
    Matrix* A = createMatrix(3, 3, A_data);
    Matrix* B = createMatrix(3, 4, B_data);

    // test batch creation
    float** C_data = (float**)malloc(sizeof(float*) * 20);
    for (i = 0; i < 20; i++){
        C_data[i] = (float*)malloc(sizeof(float) * 2);
        for (j = 0; j < 2; j++){
            C_data[i][j] = 2;
        }
    }
    DataSet* C = createDataSet(20, 2, C_data);
    DataSet** batches = createBatches(C, 6);
    for (i = 0; i < 6; i++){
        assert(batches[i]->rows == 3 + (i < 2 ? 1 : 0));
    }

    // test transpose
    Matrix* transposed = transpose(B);
    assert(transposed->rows == 4 && transposed->cols == 3);
    assert(get(transposed, 0, 1) == get(B, 1, 0));

    // test addition
    Matrix* sum = add(A, A);
    assert(sum != NULL);
    assert(sum->rows == A->rows);
    assert(sum->cols == A->cols);
    assert(get(sum, 0, 0) == 0);
    assert(get(sum, 2, 2) == 8);

    // test multiplication
    Matrix* product = multiply(A, B);
    assert(product != NULL);
    assert(product->rows == A->rows);
    assert(product->cols == B->cols);
    assert(get(product, 0, 0) == 5);
    assert(get(product, 2, 3) == 38);

    // test hadamard
    Matrix* hadamardProduct = hadamard(A, A);
    assert(get(hadamardProduct, 1, 2) == get(A, 1, 2) * get(A, 1, 2));

    // test copy
    Matrix* copied = copy(A);
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < A->rows; j++){
            assert(get(copied, i, j) == get(A, i, j));
        }
    }

    // test destroy
    destroyMatrix(A);
    destroyMatrix(B);
    destroyDataSet(C);
    for (i = 0; i < 6; i++){
        free(batches[i]);
    }
    free(batches);
    destroyMatrix(transposed);
    destroyMatrix(sum);
    destroyMatrix(product);
    destroyMatrix(hadamardProduct);
    destroyMatrix(copied);

    return 0;
}