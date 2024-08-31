#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_SENTENCE_LENGTH 512
#define MATRIX_SIZE 2 // Size of K, Q, V matrices (2x2 in this case)

// FUNCTION TO PERFORM MATRIX MULTIPLICATION
void matrix_multiply(double A[][MATRIX_SIZE], double B[][MATRIX_SIZE], double result[][MATRIX_SIZE], int rowsA, int colsA, int colsB) {

    // INITIALIZE THE RESULT MATRIX TO ZERO
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i][j] = 0.0;
        }
    }

    // PERFORM MATRIX MULTIPLICATION
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// FUNCTION TO APPLY SOFTMAX TO A MATRIX ROW
void apply_softmax(double matrix[][MATRIX_SIZE], int rows, int cols) {

    // APPLY SOFTMAX TO EACH ROW
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;

        // CALCULATE SUM OF EXPONENTS OF EACH ELEMENT IN THE ROW
        for (int j = 0; j < cols; j++) {
            sum += exp(matrix[i][j]);
        }

        // APPLY SOFTMAX NORMALIZATION
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = exp(matrix[i][j]) / sum;
        }
    }
}

// FUNCTION TO COMPUTE SELF-ATTENTION MATRIX
void compute_self_attention(double embedding_matrix[][MATRIX_SIZE], double k_matrix[][MATRIX_SIZE], double q_matrix[][MATRIX_SIZE], double v_matrix[][MATRIX_SIZE], int length, double self_attention_matrix[][MATRIX_SIZE]) {

    double qk_transpose[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0}; // RESULT MATRIX Q * K^T
    double scale_factor = 1.0 / sqrt(MATRIX_SIZE); // SCALING FACTOR (1/SQRT(DIMENSION))

    // TRANSPOSE K MATRIX FOR Q * K^T OPERATION
    double k_transpose[MATRIX_SIZE][MATRIX_SIZE] = {0};
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            k_transpose[i][j] = k_matrix[j][i];
        }
    }

    // MULTIPLY Q WITH K^T (Q * K^T)
    matrix_multiply(q_matrix, k_transpose, qk_transpose, length, MATRIX_SIZE, MATRIX_SIZE);

    // SCALE THE RESULT
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            qk_transpose[i][j] *= scale_factor;
        }
    }

    // APPLY SOFTMAX TO EACH ROW OF QK^T
    apply_softmax(qk_transpose, length, MATRIX_SIZE);

    // MULTIPLY SOFTMAX RESULT BY V MATRIX (FINAL SELF-ATTENTION MATRIX)
    matrix_multiply(qk_transpose, v_matrix, self_attention_matrix, length, MATRIX_SIZE, MATRIX_SIZE);
}

// MAIN FUNCTION FOR DEMONSTRATION
int main() {

    // DECLARE MATRICES
    double embedding_matrix[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {{1.0, 2.0}, {3.0, 4.0}}; // EXAMPLE EMBEDDING MATRIX
    double k_matrix[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 0.0}, {0.0, 1.0}}; // EXAMPLE K MATRIX
    double q_matrix[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 0.0}, {0.0, 1.0}}; // EXAMPLE Q MATRIX
    double v_matrix[MATRIX_SIZE][MATRIX_SIZE] = {{1.0, 0.0}, {0.0, 1.0}}; // EXAMPLE V MATRIX

    double self_attention_matrix[MAX_SENTENCE_LENGTH][MATRIX_SIZE] = {0}; // RESULTANT SELF-ATTENTION MATRIX

    // CALL FUNCTION TO COMPUTE SELF-ATTENTION MATRIX
    compute_self_attention(embedding_matrix, k_matrix, q_matrix, v_matrix, MAX_SENTENCE_LENGTH, self_attention_matrix);

    // PRINT THE SELF-ATTENTION MATRIX
    printf("Self-Attention Matrix:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%lf ", self_attention_matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
