#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// DEFINE VOCABULARY SIZE AND EMBEDDING DIMENSION
#define VOCAB_SIZE 1000
#define EMBEDDING_DIM 512

// FUNCTION TO INITIALIZE THE EMBEDDING MATRIX
void initialize_embedding(float embedding[VOCAB_SIZE][EMBEDDING_DIM]) {
    printf("Initializing Token Embedding Matrix:\n\n");

    // Initialize embeddings with random values
    for (int i = 0; i < VOCAB_SIZE; i++) {
        printf("Embedding vector for token index %d: [", i);
        
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            // RANDOMLY INITIALIZE EMBEDDING VECTOR FOR EACH TOKEN
            embedding[i][j] = ((float) rand() / (float)(RAND_MAX)) - 0.5; // VALUES BETWEEN -0.5 AND 0.5
            
            // PRINTING INITIALIZED VALUE
            printf("%f", embedding[i][j]);
            if (j < EMBEDDING_DIM - 1) {
                printf(", ");
            }
        }
        
        printf("]\n\n");  // NEWLINE FOR EACH TOKEN'S EMBEDDING VECTOR
    }

    printf("Token Embedding Matrix Initialized!\n\n");
}

// FUNCTION TO GENERATE POSITIONAL ENCODING
void generate_positional_encoding(float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM]) {
    printf("Generating Positional Encoding:\n\n");

    for (int pos = 0; pos < VOCAB_SIZE; pos++) {
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            if (i % 2 == 0) {
                positional_encoding[pos][i] = sin(pos / pow(10000, (2.0 * i / EMBEDDING_DIM)));  // Even index: sine
            } else {
                positional_encoding[pos][i] = cos(pos / pow(10000, (2.0 * (i - 1) / EMBEDDING_DIM)));  // Odd index: cosine
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

    // COPY THE EMBEDDING VECTOR FOR THE GIVEN TOKEN INDEX TO THE OUTPUT ARRAY
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        output[i] = embedding[token_index][i];
        printf("output[%d] = %f\n", i, output[i]);  // PRINTING EACH VALUE AS IT IS COPIED
    }

    printf("\nEmbedding Lookup Completed!\n\n");
}

// FUNCTION TO CONCATENATE TOKEN EMBEDDING AND POSITIONAL EMBEDDING
void concatenate_embeddings(float token_embedding[EMBEDDING_DIM], float positional_embedding[EMBEDDING_DIM], float *output) {
    printf("Concatenating Token and Positional Embedding:\n\n");
    
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        output[i] = token_embedding[i] + positional_embedding[i];  // Simple addition for demonstration
        printf("Concatenated output[%d] = %f\n", i, output[i]);
    }

    printf("\nConcatenation Completed!\n\n");
}

int main() {
    // DECLARE EMBEDDING MATRICES
    float embedding[VOCAB_SIZE][EMBEDDING_DIM];
    float positional_encoding[VOCAB_SIZE][EMBEDDING_DIM];

    // INITIALIZE TOKEN EMBEDDING MATRIX
    initialize_embedding(embedding);

    // GENERATE POSITIONAL ENCODING
    generate_positional_encoding(positional_encoding);

    // INPUT TOKEN INDEX (EXAMPLE: INDEX 5)
    int token_index = 5;

    // ARRAY TO STORE THE EMBEDDING VECTOR OF THE INPUT TOKEN
    float token_embedding[EMBEDDING_DIM];
    float positional_embedding[EMBEDDING_DIM];

    // LOOKUP THE EMBEDDING VECTOR FOR THE INPUT TOKEN
    lookup_embedding(embedding, token_index, token_embedding);
    
    // LOOKUP THE POSITIONAL EMBEDDING FOR THE INPUT TOKEN
    lookup_embedding(positional_encoding, token_index, positional_embedding);

    // ARRAY TO STORE THE FINAL CONCATENATED EMBEDDING
    float final_embedding[EMBEDDING_DIM];

    // CONCATENATE TOKEN EMBEDDING AND POSITIONAL EMBEDDING
    concatenate_embeddings(token_embedding, positional_embedding, final_embedding);

    // PRINT THE FINAL EMBEDDING VECTOR
    printf("Final Concatenated Embedding Vector for Token Index %d:\n\n", token_index);
    printf("[");
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        printf("%f", final_embedding[i]);
        if (i < EMBEDDING_DIM - 1) {
            printf(", ");
        }
    }
    printf("]\n\n");

    return 0;
}
