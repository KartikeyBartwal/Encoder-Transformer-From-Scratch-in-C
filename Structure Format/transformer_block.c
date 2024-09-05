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
#define MAX_SENTENCE_LENGTH 512
#define CLIP_THRESHOLD 100

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


// // Function to multiply two matrices
// void matrix_multiply(const float A[MATRIX_SIZE][EMBEDDING_DIM],
//                      const float B[EMBEDDING_DIM][EMBEDDING_DIM],
//                      float result[MATRIX_SIZE][EMBEDDING_DIM]) {
//     int i, j, k;

//     // Initialize result matrix to 0
//     for (i = 0; i < MATRIX_SIZE; i++) {
//         for (j = 0; j < EMBEDDING_DIM; j++) {
//             result[i][j] = 0.0;
//         }
//     }

//     // Perform matrix multiplication
//     for (i = 0; i < MATRIX_SIZE; i++) {
//         for (j = 0; j < EMBEDDING_DIM; j++) {
//             for (k = 0; k < EMBEDDING_DIM; k++) {
//                 result[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
// }

// FUNCTION TO PERFORM MATRIX MULTIPLICATION
void matrix_multiply(double A[][MATRIX_SIZE], double B[][MATRIX_SIZE], double result[][MATRIX_SIZE], int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i][j] = 0;
            for (int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// FUNCTION TO APPLY SOFTMAX TO A MATRIX ROW
void apply_softmax(double matrix[][MATRIX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double max_val = matrix[i][0];
        for (int j = 1; j < cols; j++) {
            if (matrix[i][j] > max_val) {
                max_val = matrix[i][j];
            }
        }

        double sum_exp = 0.0;
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = exp(matrix[i][j] - max_val); // STABILIZE WITH MAX VALUE
            sum_exp += matrix[i][j];
        }

        for (int j = 0; j < cols; j++) {
            matrix[i][j] /= sum_exp; // NORMALIZE
        }
    }
}

double clip_gradient_transformer(double gradient) {
    if (fabs(gradient) > CLIP_THRESHOLD) {
        return (gradient > 0 ? CLIP_THRESHOLD : -CLIP_THRESHOLD);
    }
    return gradient;
}

// FUNCTION TO COMPUTE SELF-ATTENTION MATRIX
void compute_self_attention(float embedding_matrix[][MATRIX_SIZE], double k_matrix[][MATRIX_SIZE], double q_matrix[][MATRIX_SIZE], double v_matrix[][MATRIX_SIZE], int length, double self_attention_matrix[][MATRIX_SIZE]) {
    double attention_scores[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0}; // TEMP MATRIX TO STORE ATTENTION SCORES

    // COMPUTE ATTENTION SCORES BY MULTIPLYING Q MATRIX WITH TRANSPOSE OF K MATRIX
    matrix_multiply(q_matrix, k_matrix, attention_scores, length, MATRIX_SIZE, MATRIX_SIZE);

    // APPLY SOFTMAX TO THE ATTENTION SCORES
    apply_softmax(attention_scores, length, MATRIX_SIZE);

    // COMPUTE SELF-ATTENTION MATRIX BY MULTIPLYING ATTENTION SCORES WITH V MATRIX
    matrix_multiply(attention_scores, v_matrix, self_attention_matrix, length, MATRIX_SIZE, MATRIX_SIZE);
}


void add_matrices(float matrix1[][MATRIX_SIZE], double matrix2[][MATRIX_SIZE], double result_matrix[][MATRIX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}



void update_attention_matrices(double loss, double learning_rate) {

    // UPDATE K MATRIX
    printf("Updating K Matrix:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // Example gradient calculation; adjust as needed
            double gradient = loss * k_matrix[i][j];
            gradient = clip_gradient_transformer(gradient); // CLIP THE GRADIENT
            k_matrix[i][j] -= learning_rate * gradient;

            // PRINT UPDATED ELEMENT FOR VERIFICATION
            printf("k_matrix[%d][%d] = %lf\n", i, j, k_matrix[i][j]);
        }
    }
    printf("\n");

    // UPDATE Q MATRIX
    printf("Updating Q Matrix:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // Example gradient calculation; adjust as needed
            double gradient = loss * q_matrix[i][j];
            gradient = clip_gradient_transformer(gradient); // CLIP THE GRADIENT
            q_matrix[i][j] -= learning_rate * gradient;

            // PRINT UPDATED ELEMENT FOR VERIFICATION
            printf("q_matrix[%d][%d] = %lf\n", i, j, q_matrix[i][j]);
        }
    }
    printf("\n");

    // UPDATE V MATRIX
    printf("Updating V Matrix:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            // Example gradient calculation; adjust as needed
            double gradient = loss * v_matrix[i][j];
            gradient = clip_gradient_transformer(gradient); // CLIP THE GRADIENT
            v_matrix[i][j] -= learning_rate * gradient;

            // PRINT UPDATED ELEMENT FOR VERIFICATION
            printf("v_matrix[%d][%d] = %lf\n", i, j, v_matrix[i][j]);
        }
    }
    printf("\n");
}
