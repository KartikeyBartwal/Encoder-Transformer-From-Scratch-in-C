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
#define EMBEDDING_DIM 2

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

// FUNCTION TO COMPUTE THE DOT PRODUCT OF TWO MATRICES
void dot_product(double A[MATRIX_SIZE][MATRIX_SIZE], double B[MATRIX_SIZE][MATRIX_SIZE], double result[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            result[i][j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// FUNCTION TO TRANSPOSE A MATRIX
void transpose(double matrix[MATRIX_SIZE][MATRIX_SIZE], double transposed[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

// FUNCTION TO APPLY SOFTMAX TO A MATRIX
void softmax(double matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        double sum_exp = 0.0;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            sum_exp += exp(matrix[i][j]);
        }
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i][j] = exp(matrix[i][j]) / sum_exp;
        }
    }
}

// FUNCTION TO CALCULATE ATTENTION SCORES
void calculate_attention(double Q[MATRIX_SIZE][MATRIX_SIZE], double K[MATRIX_SIZE][MATRIX_SIZE], double V[MATRIX_SIZE][MATRIX_SIZE], double result[MATRIX_SIZE][MATRIX_SIZE]) {
    double K_transposed[MATRIX_SIZE][MATRIX_SIZE];
    double QK_product[MATRIX_SIZE][MATRIX_SIZE];

    // TRANSPOSE THE K MATRIX
    transpose(K, K_transposed);

    // COMPUTE QK^T
    dot_product(Q, K_transposed, QK_product);

    // SCALE BY 1 / SQRT(d_k)
    double scale_factor = 1.0 / sqrt((double)MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            QK_product[i][j] *= scale_factor;
        }
    }

    // APPLY SOFTMAX TO THE RESULTING MATRIX
    softmax(QK_product);

    // COMPUTE FINAL ATTENTION OUTPUT: SOFTMAX(QK^T) * V
    dot_product(QK_product, V, result);
}


// Function to multiply two matrices
void matrix_multiply(const float A[MATRIX_SIZE][EMBEDDING_DIM],
                     const float B[EMBEDDING_DIM][EMBEDDING_DIM],
                     float result[MATRIX_SIZE][EMBEDDING_DIM]) {
    int i, j, k;

    // Initialize result matrix to 0
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < EMBEDDING_DIM; j++) {
            result[i][j] = 0.0;
        }
    }

    // Perform matrix multiplication
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < EMBEDDING_DIM; j++) {
            for (k = 0; k < EMBEDDING_DIM; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
