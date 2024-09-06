#include "Data_Preprocessing.h"

#include <stdlib.h>  // FOR MALLOC AND FREE
#include <stdio.h>
#include <math.h>    // FOR SQRT

#define MATRIX_SIZE 2
#define EMBEDDING_SIZE 512

void scale_matrix(float matrix[ EMBEDDING_SIZE][MATRIX_SIZE]) {
    float min_val = matrix[0][0];
    float max_val = matrix[0][0];

    // Find the minimum and maximum values in the matrix
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (matrix[i][j] < min_val) {
                min_val = matrix[i][j];
            }
            if (matrix[i][j] > max_val) {
                max_val = matrix[i][j];
            }
        }
    }

    // Handle case where all values are the same
    if (min_val == max_val) {
        // If min and max are the same, all values are equal, so we can't scale them
        printf("All values in the matrix are the same. Scaling is not possible.\n");
        return;
    }

    // Scale the matrix values to the range [-1, 1]
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = 2 * (matrix[i][j] - min_val) / (max_val - min_val) - 1;
        }
    }
}

void min_max_normalize_and_scale(float* data, size_t size, float new_min, float new_max) {

    if (data == NULL || size == 0) {
        return; // INVALID INPUT
    }

    // Find the minimum and maximum values in the data
    float old_min = FLT_MAX;
    float old_max = -FLT_MAX;

    for (size_t i = 0; i < size; i++) {
        if (data[i] < old_min) {
            old_min = data[i];
        }
        if (data[i] > old_max) {
            old_max = data[i];
        }
    }

    // Avoid division by zero if all values are the same
    if (old_max == old_min) {
        for (size_t i = 0; i < size; i++) {
            data[i] = new_min; // All values are the same, scale to new_min
        }
        return;
    }

    // Normalize to [0, 1]
    for (size_t i = 0; i < size; i++) {
        data[i] = (data[i] - old_min) / (old_max - old_min);
    }

    // Scale to the new range [new_min, new_max]
    for (size_t i = 0; i < size; i++) {
        data[i] = new_min + (data[i] * (new_max - new_min));

        // MAKE SURE data[ i ] HOLDS AT LEAST SOME VALUE

        if( data[ i ] == 0 ) {

            data[ i ] = 0.01;
        }

    }
}

// Function to find the length of meaningful data
size_t get_meaningful_length(const float* data, size_t size) {
    size_t start = 0;
    size_t end = size;

    // Find the start of meaningful data
    while (start < size && data[start] == 0) {
        start++;
    }

    // If all data is zero
    if (start == size) {
        return 0;
    }

    // Find the end of meaningful data
    end = size;
    while (end > start && data[end - 1] == 0) {
        end--;
    }

    return end - start;
}

void Add_Positional_Encoding(float embedding_matrix[][2], int max_sentence_length) {

    for (int i = 0; i < max_sentence_length; i++) {

        // Only apply positional encoding if the row is not all zeros
        if (!(embedding_matrix[i][0] == 0.0f && embedding_matrix[i][1] == 0.0f)) {

            // Calculate the positional encoding values
            float sin_value = sin(i / 2.0f);
            float cos_value = cos(i / 2.0f);

            // Add positional encoding to the embedding matrix
            embedding_matrix[i][0] += sin_value;
            embedding_matrix[i][1] += cos_value;

        }

    }

}
