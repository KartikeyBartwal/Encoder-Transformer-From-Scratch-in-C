#ifndef READ_WEIGHTS_H
#define READ_WEIGHTS_H

#include <stdio.h>

// Define the number of weights (should match the number of files)
#define NUM_WEIGHTS 512

/**
 * Reads weights from files and stores them in an array.
 *
 * @param path The directory path where the weight files are located.
 * @param semi_final_layer_weights The array to store the weights.
 */

void read_weights(const char* path, double* semi_final_layer_weights, int num_weights);


#endif // READ_WEIGHTS_H
