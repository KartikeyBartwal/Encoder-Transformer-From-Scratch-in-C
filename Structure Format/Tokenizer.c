#include <stdio.h>
#include <stdlib.h> // FOR MEMORY ALLOCATION
#include <string.h> // FOR STRING HANDLING FUNCTIONS
#include <ctype.h>  // FOR CHARACTER HANDLING FUNCTIONS (E.G. , TOLOWER )

#define TABLE_SIZE 100000  // SIZE OF THE HASH TABLE

// STRUCTURE FOR THE HASH TABLE NODE
typedef struct word_token_node {

    char *word;              // THE WORD
    unsigned int token_id;   // THE UNIQUE TOKEN ID
    struct word_token_node *next; // POINTER TO THE NEXT NODE IN THE LINKED LIST

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

    word_token_node* newNode = (word_token_node*)malloc(sizeof(word_token_node)); // CHANGED NODE TO WORD_TOKEN_NODE

    newNode->word = strdup(word);

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


// FUNCTION TO PRINT ALL UNIQUE WORDS
void Print_Tokens_And_Ids() {

    printf("TOKEN   VS  TOKEN_ID  :\n");

    printf("Table Size: %d \n", TABLE_SIZE);

    for (int i = 0; i < TABLE_SIZE; i++) {

        word_token_node* current = hashTable[i]; // CHANGED NODE TO WORD_TOKEN_NODE

        while (current) {

            printf("%s : %d \n", current->word , current->token_id);

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


// FUNCTION TO EXTRACT UNIQUE WORDS FROM A LIST OF SENTENCES
void extractUniqueWords(char** sentences) {

    char delimiters[] = " ,.;!?-";  // DELIMITERS FOR SPLITTING WORDS

    for (int i = 0; sentences[i] != NULL; i++) {

        char* sentence = strdup(sentences[i]); // DUPLICATE SENTENCE TO AVOID MODIFYING THE ORIGINAL

        char* token = strtok(sentence, delimiters);

        while (token != NULL) {

            if (!isWordPresent(token)) {

                insertWord(token);

            }

            token = strtok(NULL, delimiters);

        }

        free(sentence);

    }

}
