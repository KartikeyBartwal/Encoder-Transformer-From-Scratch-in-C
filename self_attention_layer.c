#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EMBEDDING_DIM 512
#define MAX_SEQ_LENGTH 128

// FUNCTION TO COMPUTE DOT PRODUCT
float dot_product(float *a, float *b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// FUNCTION TO COMPUTE SOFTMAX
void softmax(float *input, float *output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max_val); // Subtract max for numerical stability
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum; // Normalize to get probabilities
    }
}

// FUNCTION TO COMPUTE SELF-ATTENTION
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    int seq_length) {
    float attention_scores[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};
    float attention_weights[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};

    printf("Computing Attention Scores:\n");
    // COMPUTE ATTENTION SCORES
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            attention_scores[i][j] = dot_product(input[i], input[j], EMBEDDING_DIM);
            printf("Attention Score [%d][%d]: %f\n", i, j, attention_scores[i][j]);
        }
    }

    printf("\nComputing Attention Weights using Softmax:\n");
    // COMPUTE ATTENTION WEIGHTS USING SOFTMAX
    for (int i = 0; i < seq_length; i++) {
        softmax(attention_scores[i], attention_weights[i], seq_length);
        printf("Attention Weights for input %d: [", i);
        for (int j = 0; j < seq_length; j++) {
            printf("%f", attention_weights[i][j]);
            if (j < seq_length - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }

    printf("\nComputing Output of Self-Attention:\n");
    // COMPUTE OUTPUT OF SELF-ATTENTION
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            output[i][j] = 0.0f;
            for (int k = 0; k < seq_length; k++) {
                output[i][j] += attention_weights[i][k] * input[k][j];
            }
        }
    }

    printf("\nOutput of Self-Attention Layer:\n");
    for (int i = 0; i < seq_length; i++) {
        printf("Output[%d]: [", i);
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            printf("%f", output[i][j]);
            if (j < EMBEDDING_DIM - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

// MAIN FUNCTION FOR TESTING
int main() {
    // EXAMPLE SEQUENCE LENGTH
    int seq_length = 4;

    // INPUT EMBEDDINGS (FOR TESTING PURPOSES)
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {
        {0.1, 0.2, 0.3, /* ... */ 0.512},
        {0.4, 0.5, 0.6, /* ... */ 0.512},
        {0.7, 0.8, 0.9, /* ... */ 0.512},
        {0.1, 0.4, 0.7, /* ... */ 0.512}
    };

    // OUTPUT EMBEDDINGS
    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};

    // CALL SELF-ATTENTION FUNCTION
    self_attention(input, output, seq_length);

    return 0;
}
