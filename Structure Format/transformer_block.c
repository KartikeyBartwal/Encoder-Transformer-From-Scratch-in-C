#include "transformer_block.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// FUNCTION IMPLEMENTATION FOR POSITIONAL ENCODING

double* positional_encoding( int index, int vector_size ) {
    // ALLOCATE MEMORY FOR THE RETURN ARRAY
   double* encoding = ( double* ) malloc( vector_size * sizeof( double ) );
  // CALCULATE EACH VALUE IN THE ENCODING VECTOR
  for( int i = 0; i < vector_size; i++ ) {
      if( i % 2 == 0 ) {
          // EVEN INDEX
         encoding[ i ] = sin( index / pow( 10000.0 , ( double ) i / vector_size ));
      }
      else{
          // ODD INDEX
         encoding[ i ] = cos( index / pow( 10000.0 , ( double ) (i - 1) / vector_size));
      }
  }
  return encoding;
}


/////////////////////// SELF ATTENTION ////////////////////////////

#define MATRIX_SIZE 2

// DEFINE MATRICES

double k_matrix[ MATRIX_SIZE ][ MATRIX_SIZE ];
double q_matrix[ MATRIX_SIZE ][ MATRIX_SIZE ];
double v_matrix[ MATRIX_SIZE ][ MATRIX_SIZE ];

// FUNCTION TO READ A SINGLE VALUE FROM A FILE
double read_single_value_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");

    printf("filename: %s \n" , filename);

    if (file == NULL) {
        perror("Error opening file");
        return 0.0;
    }

    double value;
    if (fscanf(file, "%lf", &value) != 1) {
        perror("Error reading file");
        fclose(file);
        return 0.0;
    }

    fclose(file);
    return value;
}


// FUNCTION TO INITIALIZE MATRICES WITH WEIGHTS FROM FILES

// FUNCTION TO INITIALIZE MATRICES WITH WEIGHTS FROM FILES
void initialize_matrices_from_files() {
    int index = 0;

    // READ KEY MATRICES
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            char filename[256];

            printf( "key_weight_%d.txt \n" , index + 1);

            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/key_weight_%d.txt", index + 1);
            k_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }

    printf("Initialized KEY MATRIX\n");

    index = 0;

    // READ QUERY MATRICES
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            char filename[256];

            printf( "query_weight_%d.txt \n" , index + 1);

            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/query_weight_%d.txt", index + 1);
            q_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }

    printf("Initialized QUERY MATRIX\n");

    index = 0;

    // READ VALUE MATRICES
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            char filename[256];

            printf( "value_weight_%d.txt \n" , index + 1);
            snprintf(filename, sizeof(filename), "Model Trained Weights/self-attention-block-weights/value_weight_%d.txt", index + 1);
            v_matrix[i][j] = read_single_value_from_file(filename);
            index++;
        }
    }

    printf("Initialized VALUE MATRIX\n");
}

// FUNCTION TO PRINT THE MATRICES
void print_matrix(const char* name, double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
        printf("%s:\n", name);
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%.2f ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("\n");
}
