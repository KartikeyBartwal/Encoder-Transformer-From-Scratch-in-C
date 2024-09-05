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
    float embedding[ 20 ];  // THE 2-VALUE WORD EMBEDDING

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

    // PRINT HEADER FOR OUTPUT WITH BETTER ALIGNMENT
    printf("------------------------------------------------------------\n");
    printf("| %-20s | %-10s | %-30s |\n", "TOKEN", "TOKEN_ID", "EMBEDDING");
    printf("------------------------------------------------------------\n");

    // ITERATE OVER THE HASH TABLE
    for (int i = 0; i < TABLE_SIZE; i++) {

        word_token_node* current = hashTable[i]; // CHANGED NODE TO WORD_TOKEN_NODE

        // TRAVERSE THE LINKED LIST AT EACH HASH INDEX
        while (current) {

            // PRINT WORD AND TOKEN ID WITH ALIGNMENT
            printf("| %-20s | %-10d | { ", current->word, current->token_id);

            // PRINT ALL 20 ELEMENTS OF THE EMBEDDING
            for (int j = 0; j < 20; j++) {

                printf("%.4f", current->embedding[j]);

                // ADD A COMMA BETWEEN ELEMENTS, EXCEPT AFTER THE LAST ELEMENT
                if (j < 19) {
                    printf(", ");
                }
            }

            printf(" } |\n"); // CLOSE THE EMBEDDING BRACKET AND COLUMN
            printf("------------------------------------------------------------\n");

            current = current->next; // MOVE TO THE NEXT NODE
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

    // DEFINE THE SIZE OF THE EMBEDDING
    int size = 20;

    // ALLOCATE MEMORY FOR A 2D ARRAY (20 X 1)
    float** embedding = malloc(size * sizeof(float*));

    // CHECK IF MEMORY ALLOCATION SUCCEEDED
    if (embedding == NULL) {

        printf("Memory allocation failed for embedding.\n");

        return NULL;
    }

    // ALLOCATE MEMORY FOR EACH ROW
    for (int i = 0; i < size; i++) {
        embedding[i] = malloc(sizeof(float));

        // CHECK IF MEMORY ALLOCATION SUCCEEDED FOR EACH ROW
        if (embedding[i] == NULL) {

            printf("Memory allocation failed for embedding[%d].\n", i);

            // FREE PREVIOUSLY ALLOCATED MEMORY
            for (int j = 0; j < i; j++) {
                free(embedding[j]);
            }
            free(embedding);

            return NULL;
        }
    }

    // GENERATE RANDOM PARAMETERS AND COMPUTE EMBEDDING VALUES
    for (int i = 0; i < size; i++) {
        float param = generate_random();  // GENERATE RANDOM PARAMETER BETWEEN 0 AND 1

        // COMPUTE EMBEDDING VALUES USING SIN AND COS FUNCTIONS
        embedding[i][0] = (i % 2 == 0 ? sin(token_id) : cos(token_id)) * param;
    }

    return embedding;
}


// FUNCTION TO EXTRACT UNIQUE WORDS FROM A LIST OF SENTENCES AND GENERATE EMBEDDINGS
void extractUniqueWords(char** sentences) {

    // DEFINE DELIMITERS FOR TOKENIZATION
    char delimiters[] = " ,.;!?-";

    // ITERATE OVER EACH SENTENCE
    for (int i = 0; sentences[i] != NULL; i++) {

        // DUPLICATE THE SENTENCE TO AVOID MODIFYING THE ORIGINAL
        char* sentence = strdup(sentences[i]);

        // CHECK IF MEMORY ALLOCATION FOR THE SENTENCE SUCCEEDED
        if (sentence == NULL) {
            printf("Memory allocation failed for sentence.\n");
            continue;
        }

        // TOKENIZE THE SENTENCE USING THE DELIMITERS
        char* token = strtok(sentence, delimiters);

        while (token != NULL) {

            // CHECK IF THE WORD IS NOT ALREADY PRESENT
            if (!isWordPresent(token)) {

                insertWord(token);  // INSERT THE WORD INTO THE DATA STRUCTURE

                unsigned int token_id = getTokenId(token);  // GET TOKEN ID FOR THE WORD

                float** embedding = Word_Embedding_Generation(token_id);  // GENERATE EMBEDDING FOR THE WORD

                // CHECK IF EMBEDDING GENERATION SUCCEEDED
                if (embedding == NULL) {
                    printf("Embedding generation failed.\n");
                    continue;
                }

                // FIND THE CORRECT LOCATION IN THE HASH TABLE
                unsigned int index = hash(token);
                word_token_node* current = hashTable[index];

                // TRAVERSE THE LINKED LIST TO FIND THE WORD TOKEN NODE
                while (current) {
                    if (strcmp(current->word, token) == 0) {
                        // COPY ALL 20 ELEMENTS OF THE EMBEDDING
                        for (int j = 0; j < 20; j++) {
                            current->embedding[j] = embedding[j][0];
                        }
                        break;
                    }
                    current = current->next;
                }

                // FREE THE MEMORY ALLOCATED FOR EMBEDDING
                for (int j = 0; j < 20; j++) {
                    free(embedding[j]);
                }
                free(embedding);
            }

            // GET THE NEXT TOKEN
            token = strtok(NULL, delimiters);
        }

        free(sentence);  // FREE THE DUPLICATED SENTENCE
    }
}

// FUNCTION TO GET EMBEDDING FOR A GIVEN TOKEN ID AND RETURN AS DOUBLE ARRAY OF 2 ELEMENTS
void getEmbeddingByTokenId(unsigned int token_id, double expected_embedding[20]) {

    // ITERATE OVER THE ENTIRE HASH TABLE
    for (int i = 0; i < TABLE_SIZE; i++) {

        word_token_node* current = hashTable[i];

        // TRAVERSE THE LINKED LIST AT EACH HASH INDEX
        while (current) {

            // CHECK IF THE TOKEN ID MATCHES
            if (current->token_id == token_id) {

                // ASSIGN EMBEDDING VALUES TO OUTPUT ARRAY
                for (int j = 0; j < 20; j++) {

                    expected_embedding[j] = (double)current->embedding[j];
                }

                return;  // RETURN IF SUCCESSFUL
            }

            current = current->next;
        }
    }

    // TOKEN ID NOT FOUND, RETURN
    return;
}
