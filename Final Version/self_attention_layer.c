#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VOCAB_SIZE 1000
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

// FUNCTION TO INITIALIZE THE EMBEDDING MATRIX
void initialize_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM]) {
    printf("Initializing Token Embedding Matrix:\n\n");

    for (int i = 0; i < VOCAB_SIZE; i++) {
        printf("Embedding vector for token index %d: [", i);
        
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            embedding[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5; // VALUES BETWEEN -0.5 AND 0.5
            
            printf("%f", embedding[i][j]);
            if (j < EMBEDDING_DIM - 1) {
                printf(", ");
            }
        }
        
        printf("]\n\n");  
    }

    printf("Token Embedding Matrix Initialized!\n\n");
}

// FUNCTION TO GENERATE POSITIONAL ENCODING
void generate_positional_encoding(float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM]) {
    printf("Generating Positional Encoding:\n\n");

    for (int pos = 0; pos < VOCAB_SIZE; pos++) {
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            if (i % 2 == 0) {
                positional_encoding[pos][i] = sin(pos / pow(10000, (2.0 * i / EMBEDDING_DIM)));  
            } else {
                positional_encoding[pos][i] = cos(pos / pow(10000, (2.0 * (i - 1) / EMBEDDING_DIM)));  
            }
        }
        printf("Positional Encoding for position %d: [", pos);
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            printf("%f", positional_encoding[pos][j]);
            if (j < EMBEDDING_DIM - 1) {
                printf(", ");
            }
        }
        printf("]\n\n");
    }

    printf("Positional Encoding Generated!\n\n");
}

// FUNCTION TO LOOKUP EMBEDDING FOR A GIVEN TOKEN INDEX
void lookup_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM], int token_index, float *output) {
    printf("Looking up Embedding for Token Index %d:\n\n", token_index);

    for (int i = 0; i < EMBEDDING_DIM; i++) {
        output[i] = embedding[token_index][i];
        printf("output[%d] = %f\n", i, output[i]);
    }

    printf("\nEmbedding Lookup Completed!\n\n");
}

// FUNCTION TO CONCATENATE TOKEN EMBEDDING AND POSITIONAL EMBEDDING
void concatenate_embeddings(float token_embedding[EMBEDDING_DIM], float positional_embedding[EMBEDDING_DIM], float *output) {
    printf("Concatenating Token and Positional Embedding:\n\n");
    
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        output[i] = token_embedding[i] + positional_embedding[i];  
        printf("Concatenated output[%d] = %f\n", i, output[i]);
    }

    printf("\nConcatenation Completed!\n\n");
}

// FUNCTION TO MULTIPLY TWO MATRICES
void matrix_multiply(float A[MAX_SEQ_LENGTH][EMBEDDING_DIM], float B[EMBEDDING_DIM][EMBEDDING_DIM], float C[MAX_SEQ_LENGTH][EMBEDDING_DIM], int rows_A, int cols_A, int cols_B) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// FUNCTION TO INITIALIZE WEIGHT MATRICES
void initialize_weight_matrix(float weight[EMBEDDING_DIM][EMBEDDING_DIM]) {
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            weight[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5; // Random values between -0.5 and 0.5
        }
    }
}

// FUNCTION TO COMPUTE SELF-ATTENTION WITH TRAINABLE K, Q, V
void self_attention(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                    int seq_length) {
    float W_Q[EMBEDDING_DIM][EMBEDDING_DIM]; // Weight matrix for Q
    float W_K[EMBEDDING_DIM][EMBEDDING_DIM]; // Weight matrix for K
    float W_V[EMBEDDING_DIM][EMBEDDING_DIM]; // Weight matrix for V

    // Initialize weight matrices
    initialize_weight_matrix(W_Q);
    initialize_weight_matrix(W_K);
    initialize_weight_matrix(W_V);

    float Q[MAX_SEQ_LENGTH][EMBEDDING_DIM]; // Query matrix
    float K[MAX_SEQ_LENGTH][EMBEDDING_DIM]; // Key matrix
    float V[MAX_SEQ_LENGTH][EMBEDDING_DIM]; // Value matrix

    // Compute Q, K, V by multiplying input with the weight matrices
    matrix_multiply(input, W_Q, Q, seq_length, EMBEDDING_DIM, EMBEDDING_DIM);
    matrix_multiply(input, W_K, K, seq_length, EMBEDDING_DIM, EMBEDDING_DIM);
    matrix_multiply(input, W_V, V, seq_length, EMBEDDING_DIM, EMBEDDING_DIM);

    float attention_scores[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};
    float attention_weights[MAX_SEQ_LENGTH][MAX_SEQ_LENGTH] = {0};

    // Compute attention scores
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            attention_scores[i][j] = dot_product(Q[i], K[j], EMBEDDING_DIM) / sqrt(EMBEDDING_DIM);
        }
    }

    // Compute attention weights using softmax
    for (int i = 0; i < seq_length; i++) {
        softmax(attention_scores[i], attention_weights[i], seq_length);
    }

    // Compute output of self-attention
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            output[i][j] = 0.0f;
            for (int k = 0; k < seq_length; k++) {
                output[i][j] += attention_weights[i][k] * V[k][j];
            }
        }
    }
}

void feed_forward(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], int seq_length) {
    float W1[EMBEDDING_DIM][EMBEDDING_DIM * 4]; // First layer weights
    float W2[EMBEDDING_DIM * 4][EMBEDDING_DIM]; // Second layer weights
    
    // Initialize weights
    initialize_weight_matrix(W1);
    initialize_weight_matrix(W2);
    
    // First linear transformation with ReLU
    float intermediate[MAX_SEQ_LENGTH][EMBEDDING_DIM * 4] = {0};
    matrix_multiply(input, W1, intermediate, seq_length, EMBEDDING_DIM, EMBEDDING_DIM * 4);
    
    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < EMBEDDING_DIM * 4; j++) {
            intermediate[i][j] = fmax(0, intermediate[i][j]); // ReLU activation
        }
    }
    
    // Second linear transformation
    matrix_multiply(intermediate, W2, output, seq_length, EMBEDDING_DIM * 4, EMBEDDING_DIM);
}

void layer_normalization(float input[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                         float output[MAX_SEQ_LENGTH][EMBEDDING_DIM], 
                         int seq_length) {
    for (int i = 0; i < seq_length; i++) {
        float mean = 0.0f;
        float variance = 0.0f;
        
        // Compute the mean
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            mean += input[i][j];
        }
        mean /= EMBEDDING_DIM;

        // Compute the variance
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            variance += (input[i][j] - mean) * (input[i][j] - mean);
        }
        variance /= EMBEDDING_DIM;

        // Compute the normalization
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            output[i][j] = (input[i][j] - mean) / sqrt(variance + 1e-6); // Add epsilon for numerical stability
        }
    }
}


// MAIN FUNCTION FOR TESTING
int main() {
    // DECLARE EMBEDDING MATRICES
    float embedding[VOCAB_SIZE][EMBEDDING_DIM];
    float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM];

    // INITIALIZE TOKEN EMBEDDING MATRIX
    initialize_embedding(embedding);

    // GENERATE POSITIONAL ENCODING
    generate_positional_encoding(positional_encoding);

    // INPUT SEQUENCE LENGTH
    int seq_length = 4; // Example sequence length

    // INPUT EMBEDDINGS (FOR TESTING PURPOSES)
    float input[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};

    // LOOKUP EMBEDDING FOR EACH TOKEN IN THE SEQUENCE
    for (int i = 0; i < seq_length; i++) {
        float token_embedding[EMBEDDING_DIM];
        lookup_embedding(embedding, i, token_embedding);
        
        float positional_embedding[EMBEDDING_DIM];
        lookup_embedding(positional_encoding, i, positional_embedding);

        // CONCATENATE TOKEN AND POSITIONAL EMBEDDINGS
        float final_embedding[EMBEDDING_DIM];
        concatenate_embeddings(token_embedding, positional_embedding, final_embedding);

        // STORE THE FINAL EMBEDDING IN THE INPUT ARRAY FOR SELF-ATTENTION
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            input[i][j] = final_embedding[j];
        }
    }

    // OUTPUT MATRIX FOR SELF-ATTENTION
    float output[MAX_SEQ_LENGTH][EMBEDDING_DIM] = {0};

    // CALL SELF-ATTENTION FUNCTION
    self_attention(input, output, seq_length);

    // The output is ready to be passed to subsequent layers

    return 0;
}
