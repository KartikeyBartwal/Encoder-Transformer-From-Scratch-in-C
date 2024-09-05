#include <stdio.h>
#include <stdlib.h> // FOR MEMORY ALLOCATION, rand() and RAND_MAX
#include <string.h> // FOR STRING HANDLING FUNCTIONS
#include <ctype.h>  // FOR CHARACTER HANDLING FUNCTIONS (E.G. , TOLOWER )
#include <math.h>     // FOR sin() AND cos()

#define TABLE_SIZE 100000  // SIZE OF THE HASH TABLE

// STRUCTURE FOR THE HASH TABLE NODE
typedef struct word_token_node {

    char *word;              // THE WORD
    unsigned int token_id;   // THE UNIQUE TOKEN ID
    struct word_token_node *next; // POINTER TO THE NEXT NODE IN THE LINKED LIST
    float embedding[ 2 ];  // THE 2-VALUE WORD EMBEDDING

} word_token_node;


// GLOBAL VARIABLES
word_token_node* hashTable[TABLE_SIZE] = { NULL }; // HASH TABLE FOR STORING WORD TOKENS
int global_token = 1;  // GLOBAL TOKEN ID COUNTER



// IMPROVED HASH FUNCTION USING DJB2 ALGORITHM
unsigned int hash(const char* word) {

    unsigned long hashValue = 5381;

    int c;

    while ((c = *word++)) {

        hashValue = ((hashValue << 5) + hashValue) + tolower(c); // HASH * 33 + C

    }

    return hashValue % TABLE_SIZE;

}


// FUNCTION TO INSERT WORD INTO THE HASH TABLE
void insertWord(const char* word) {

    unsigned int index = hash(word);

    word_token_node* newNode = (word_token_node*)malloc(sizeof(word_token_node));

    if (newNode == NULL) {
        printf("Memory allocation failed for newNode.\n");
        return;
    }

    newNode->word = strdup(word);

    if (newNode->word == NULL) {
        printf("Memory allocation failed for newNode->word.\n");
        free(newNode);
        return;
    }

    newNode->token_id = global_token;
    global_token += 1;
    newNode->next = hashTable[index];
    hashTable[index] = newNode;
}


// FUNCTION TO CHECK IF THE WORD IS ALREADY IN THE HASH TABLE
int isWordPresent(const char* word) {

    unsigned int index = hash(word);

    word_token_node* current = hashTable[index]; // CHANGED NODE TO WORD_TOKEN_NODE

    while (current) {

        if (strcmp(current->word, word) == 0) {

            return 1;  // WORD FOUND

        }

        current = current->next;

    }

    return 0;  // WORD NOT FOUND

}


// FUNCTION TO PRINT ALL UNIQUE WORDS, THEIR TOKEN IDS, AND EMBEDDINGS
void Print_Tokens_And_Ids() {

    printf("TOKEN   VS  TOKEN_ID   VS  EMBEDDING:\n");
    printf("Table Size: %d \n", TABLE_SIZE);

    for (int i = 0; i < TABLE_SIZE; i++) {

        word_token_node* current = hashTable[i]; // CHANGED NODE TO WORD_TOKEN_NODE

        while (current) {

            // PRINT WORD, TOKEN ID, AND EMBEDDING VALUES
            printf("%s : %d : { %.4f, %.4f }\n", current->word, current->token_id, current->embedding[0], current->embedding[1]);

            current = current->next;
        }

    }

}


// FUNCTION TO GET TOKEN ID FOR A WORD
unsigned int getTokenId(const char* word) {
    unsigned int index = hash(word);
    word_token_node* current = hashTable[index];

    while (current) {
        if (strcmp(current->word, word) == 0) {
            return current->token_id;  // WORD FOUND, RETURN ITS TOKEN ID
        }
        current = current->next;
    }

    return 0;  // WORD NOT FOUND, RETURN 0 (OR YOU COULD RETURN A SPECIAL VALUE FOR UNKNOWN WORDS)
}

// FUNCTION TO GENERATE RANDOM FLOAT BETWEEN -50 to 50
float generate_random() {

    // GENERATE RANDOM FLOAT BETWEEN 0 AND 1
    float random_fraction = (float) rand() / (float) RAND_MAX;

    // SCALE AND SHIFT TO THE RANGE [-50000, +50000]
    float min = -50.0f;
    float max = 50.0f;
    float range = max - min;

    return min + (random_fraction * range);
}

// FUNCTION TO GENERATE WORD EMBEDDING FOR A GIVEN TOKEN ID
float** Word_Embedding_Generation(int token_id) {

    // ALLOCATE MEMORY FOR A 2D ARRAY (2 X 1)
    float** embedding = malloc(2 * sizeof(float*));

    // CHECK IF MEMORY ALLOCATION SUCCEEDED
    if (embedding == NULL) {

        printf("Memory allocation failed for embedding.\n");

        return NULL;
    }

    embedding[0] = malloc(sizeof(float));

    // CHECK IF MEMORY ALLOCATION SUCCEEDED
    if (embedding[0] == NULL) {

        printf("Memory allocation failed for embedding[0].\n");

        free(embedding); // FREE PREVIOUSLY ALLOCATED MEMORY

        return NULL;
    }

    embedding[1] = malloc(sizeof(float));

    // CHECK IF MEMORY ALLOCATION SUCCEEDED
    if (embedding[1] == NULL) {

        printf("Memory allocation failed for embedding[1].\n");

        free(embedding[0]); // FREE PREVIOUSLY ALLOCATED MEMORY

        free(embedding);

        return NULL;
    }

    // GENERATE RANDOM PARAMETERS BETWEEN 0 AND 1
    float param_1 = generate_random();

    float param_2 = generate_random();

    // COMPUTE EMBEDDING VALUES
    embedding[0][0] = sin(token_id) * param_1;

    embedding[1][0] = cos(token_id) * param_2;

    return embedding;
}


// FUNCTION TO EXTRACT UNIQUE WORDS FROM A LIST OF SENTENCES AND GENERATE EMBEDDINGS
void extractUniqueWords(char** sentences) {

    char delimiters[] = " ,.;!?-";

    for (int i = 0; sentences[i] != NULL; i++) {

        char* sentence = strdup(sentences[i]);

        if (sentence == NULL) {
            printf("Memory allocation failed for sentence.\n");
            continue;
        }

        char* token = strtok(sentence, delimiters);

        while (token != NULL) {

            if (!isWordPresent(token)) {

                insertWord(token);

                unsigned int token_id = getTokenId(token);

                float** embedding = Word_Embedding_Generation(token_id);

                if (embedding == NULL) {
                    printf("Embedding generation failed.\n");
                    continue;
                }

                unsigned int index = hash(token);
                word_token_node* current = hashTable[index];

                while (current) {
                    if (strcmp(current->word, token) == 0) {
                        current->embedding[0] = embedding[0][0];
                        current->embedding[1] = embedding[1][0];
                        break;
                    }
                    current = current->next;
                }

                free(embedding[0]);
                free(embedding[1]);
                free(embedding);
            }

            token = strtok(NULL, delimiters);
        }

        free(sentence);
    }
}

// FUNCTION TO GET EMBEDDING FOR A GIVEN TOKEN ID AND RETURN AS DOUBLE ARRAY OF 2 ELEMENTS
void getEmbeddingByTokenId(unsigned int token_id, double expected_embedding[2]) {

    // ITERATE OVER THE ENTIRE HASH TABLE
    for (int i = 0; i < TABLE_SIZE; i++) {

        word_token_node* current = hashTable[i];

        // TRAVERSE THE LINKED LIST AT EACH HASH INDEX
        while (current) {

            // CHECK IF THE TOKEN ID MATCHES
            if (current->token_id == token_id) {

                // ASSIGN EMBEDDING VALUES TO OUTPUT ARRAY
                expected_embedding[0] = (double)current->embedding[0];
                expected_embedding[1] = (double)current->embedding[1];

                return ;  // RETURN 1 IF SUCCESSFUL
            }

            current = current->next;
        }
    }

    return ; // TOKEN ID NOT FOUND, RETURN
}
