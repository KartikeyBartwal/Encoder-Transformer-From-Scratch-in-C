#include <stdio.h>
#include "transformer_block.h"
#include <math.h>
#include <stdlib.h>
#include "Tokenizer.h"

int main() {
    // GENERATE EMBEDDING

    // PRINT THE EMBEDDING VALUES

    for( int token_id = 0; token_id < 100; token_id++ ) {

        float** embedding = Word_Embedding_Generation(token_id);

        printf("Word Embedding for token ID %d:\n", token_id);
        printf("{ %.4f, %.4f }\n", embedding[0][0], embedding[1][0]);

        // FREE THE ALLOCATED MEMORY
        free(embedding[0]);
        free(embedding[1]);
        free(embedding);
    }


    return 0;
}
