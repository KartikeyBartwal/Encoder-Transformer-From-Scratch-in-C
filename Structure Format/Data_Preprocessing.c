#include "Data_Preprocessing.h"

#include <stdlib.h>  // FOR MALLOC AND FREE

#include <math.h>    // FOR SQRT

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
