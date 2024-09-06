#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MATRIX_SIZE 10 // Adjust as needed

void scale_matrix(double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    double min_val = matrix[0][0];
    double max_val = matrix[0][0];

    // Find the minimum and maximum values in the matrix
    for (int i = 0; i < MATRIX_SIZE; i++) {
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

void print_matrix(double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {

    for( int row = 0; row < 512; row++ ) {

        for( int node = 0; node < 65; node++ ) {

            printf(" %d \n" , ( row * 65 ) + node );
        }
    }

    return 0;
}
