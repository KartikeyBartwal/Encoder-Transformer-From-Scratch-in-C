#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define TABLE_SIZE 5000

typedef struct word_token_node {
    char *word;

    unsigned int token_id;

    struct word_token_node *next;

} word_token_node;

extern word_token_node* hashTable[TABLE_SIZE];
extern int global_token;

// FUNCTION DECLARATIONS

// FUNCTION TO READ THE ENTIRE CONTENT OF A FILE AND RETURN IT AS A STRING
char* readFileToString(const char *filename);

// FUNCTION TO SPLIT THE TEXT INTO SENTENCES
char** SplitSentences(char *raw_text);

// FUNCTION TO CLEAN TEXT BY REMOVING UNWANTED CHARACTERS
char* Cleaned_Text(char *raw_text);

// IMPROVED HASH FUNCTION USING DJB2 ALGORITHM
unsigned int hash(const char* word);

// FUNCTION TO INSERT A WORD INTO THE HASH TABLE
void insertWord(const char* word);

// FUNCTION TO CHECK IF A WORD IS ALREADY PRESENT IN THE HASH TABLE
int isWordPresent(const char* word);

// FUNCTION TO PRINT ALL UNIQUE WORDS WITH THEIR TOKEN IDs
void Print_Tokens_And_Ids();

// FUNCTION TO EXTRACT UNIQUE WORDS FROM A LIST OF SENTENCES
void extractUniqueWords(char** sentences);

unsigned int getTokenId(const char* word);


// FOR WORD EMBEDDINGS
float generate_random();

// FUNCTION TO GENERATE WORD EMBEDDING FOR A GIVEN TOKEN ID
float** Word_Embedding_Generation( int token_id );

float* getEmbedding(unsigned int token_id);

void getEmbeddingByTokenId(unsigned int token_id, double expected_embedding[2]);

#endif /* TOKENIZER.H */
