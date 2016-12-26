// activation and error functions such as softmax, sigmoid, etc
#include "matrix.c"
#include <assert.h>
#include <math.h>
#include <float.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

double sigmoidFunc(double input);

double sigmoidDeriv(double input);

void sigmoid(Matrix* input);

void softmax(Matrix* input);

double crossEntropyLoss(Matrix* prediction, Matrix* actual);