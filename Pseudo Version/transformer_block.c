#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB_SIZE 1000
#define EMBEDDING_DIM 512
#define MAX_SEQ_LENGTH 128

// FUNCTION PROTOTYPES
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    int seq_length);

void layer_normalization(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                         float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                         int seq_length);

void feed_forward(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                  float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                  int seq_length);

// TRANSFORMER BLOCK FUNCTION
void transformer_block(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                       float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                       int seq_length) {
    // Step 1: Self-Attention
    float attention_output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    self_attention(input, attention_output, seq_length);

    // Step 2: Layer Normalization after Self-Attention
    float normed_attention_output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    layer_normalization(attention_output, normed_attention_output, seq_length);

    // Step 3: Feed Forward Network
    float ff_output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    feed_forward(normed_attention_output, ff_output, seq_length);

    // Step 4: Layer Normalization after Feed Forward Network
    layer_normalization(ff_output, output, seq_length);
}

// SELF-ATTENTION FUNCTION (As previously defined)
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    int seq_length) {
    // Implementation as previously defined
}

// LAYER NORMALIZATION FUNCTION (As previously defined)
void layer_normalization(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                         float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                         int seq_length) {
    // Implementation as previously defined
}

// FEED FORWARD NETWORK FUNCTION
void feed_forward(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                  float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                  int seq_length) {
    // Define weights for the feed-forward network
    float weight1[EMBEDDING_DIM][EMBEDDING_DIM * 4]; // First layer weights
    float weight2[EMBEDDING_DIM * 4][EMBEDDING_DIM]; // Second layer weights

    // Initialize weights (you can use random initialization or predefined weights)
    // For simplicity, initializing to random values here
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        for (int j = 0; j < EMBEDDING_DIM * 4; j++) {
            weight1[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5; // Random values
        }
    }
    
    for (int i = 0; i < EMBEDDING_DIM * 4; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            weight2[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5; // Random values
        }
    }

    // Step 1: First layer transformation
    float intermediate_output[MAX_SEQ_LENGTH][EMBEDDING_DIM * 4] = {0};
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM * 4; j++) {
            for (int k = 0; k < EMBEDDING_DIM; k++) {
                intermediate_output[i][j] += input[i][k] * weight1[k][j];
            }
        }
    }

    // Step 2: Apply activation function (ReLU)
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM * 4; j++) {
            intermediate_output[i][j] = fmaxf(0.0f, intermediate_output[i][j]); // ReLU activation
        }
    }

    // Step 3: Second layer transformation
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            for (int k = 0; k < EMBEDDING_DIM * 4; k++) {
                output[i][j] += intermediate_output[i][k] * weight2[k][j];
            }
        }
    }
}

// MAIN FUNCTION FOR TESTING
int main() {
    // Example usage of the transformer block
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0}; // Your input embeddings
    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};
    int seq_length = 4; // Example sequence length

    // Call the transformer block
    transformer_block(input, output, seq_length);

    // Output the final result
    printf("Output of Transformer Block:\n");
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

    return 0;
}
