#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <math.h>  // INCLUDE NECESSARY LIBRARIES

// FUNCTION TO CALCULATE MEAN SQUARED ERROR (MSE)
double calculate_mse(double output_array[], double expected_output_array[], int size);

void update_weights_last_layer(double loss, double LEARNING_RATE , double final_layer_weights[], double semi_final_layer_weights[], int final_layer_size, int semi_final_layer_size);

void update_semi_final_layer_weights(double loss, double LEARNING_RATE, double semi_final_layer_weights[], int semi_final_layer_size);

void update_weights_self_attention_layer( double loss );

#endif // BACKPROPAGATION_H
