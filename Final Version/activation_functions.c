#include <math.h>
#include <stdio.h>


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double leaky_relu(double x, double alpha) {
    return x > 0 ? x : alpha * x;
}

double swish(double x) {
    return x * sigmoid(x);
}
