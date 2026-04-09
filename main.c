#include <stdio.h>
#include "nn.h"

int main(void) {
    // XOR problem: 2 inputs -> 4 hidden -> 1 output
    int layers[] = {2, 4, 1};
    NN *net = nn_create(3, layers);

    // training data
    double i0[]={0,0},i1[]={0,1},i2[]={1,0},i3[]={1,1};
    double t0[]={0},t1[]={1},t2[]={1},t3[]={0};

    double *inputs[]={i0,i1,i2,i3};
    double *targets[]={t0,t1,t2,t3};

    printf("Training on XOR...\n\n");
    nn_train(net, inputs, targets, 4, 10000, 2.0);

    printf("\nResults:\n");
    double out[1];
    for (int i=0; i<4; i++) {
        nn_forward(net, inputs[i], out);
        printf("  %.0f XOR %.0f = %.4f (expected %0.f)\n",inputs[i][0],inputs[i][1],out[0],targets[i][0]);
    }
    nn_free(net);
    return 0;
}
