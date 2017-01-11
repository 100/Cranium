#include "../src/std_includes.h"
#include "../src/matrix.h"
#include "../src/function.h"
#include "../src/layer.h"

int main(){
    // test creation
    Layer* layer = createLayer(INPUT, 10, sigmoid);
    Layer* layer2 = createLayer(HIDDEN, 5, softmax);
    Connection* connection = createConnection(layer, layer2);
    assert(layer->input->rows == 1 && layer->input->cols == 10);
    assert(connection->weights->rows == 10 && connection->weights->cols == 5);
    assert(connection->bias->rows == 1 && connection->weights->cols == connection->bias->cols);

    // test connection initialization
    initializeConnection(connection);
    int i;
    for (i = 0; i < connection->bias->cols; i++){
        assert(connection->bias->data[i] == 0);
    }
    
    // test layer activation
    Layer* layer3 = createLayer(INPUT, 10, sigmoid);
    for (i = 0; i < 10; i++){
        layer3->input->data[i] = i * 2;
    }
    activateLayer(layer3);
    for (i = 0; i < 10; i++){
        assert(layer3->input->data[i] >= 0 && layer3->input->data[i] <= 1);
    }

    // test destroy
    destroyLayer(layer);
    destroyLayer(layer2);
    destroyLayer(layer3);
    destroyConnection(connection);

    return 0;
}