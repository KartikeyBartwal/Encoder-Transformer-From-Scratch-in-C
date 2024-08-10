#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define TABLE_SIZE 1000  // SIZE OF THE HASH TABLE

// STRUCTURE FOR THE HASH TABLE NODE
typedef struct Node {
    char* word;
    struct Node* next;
} Node;

// HASH TABLE TO STORE UNIQUE WORDS
Node* hashTable[TABLE_SIZE] = { NULL };

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
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->word = strdup(word);
    newNode->next = hashTable[index];
    hashTable[index] = newNode;
}

// FUNCTION TO CHECK IF THE WORD IS ALREADY IN THE HASH TABLE
int isWordPresent(const char* word) {
    unsigned int index = hash(word);
    Node* current = hashTable[index];
    while (current) {
        if (strcmp(current->word, word) == 0) {
            return 1;  // WORD FOUND
        }
        current = current->next;
    }
    return 0;  // WORD NOT FOUND
}

// FUNCTION TO PRINT ALL UNIQUE WORDS
void printUniqueWords() {
    printf("Unique words in the corpus:\n");
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node* current = hashTable[i];
        while (current) {
            printf("%s\n", current->word);
            current = current->next;
        }
    }
}

// FUNCTION TO EXTRACT UNIQUE WORDS FROM A LIST OF SENTENCES
void extractUniqueWords(char** sentences) {
    char delimiters[] = " ,.;!?-";  // DELIMITERS FOR SPLITTING WORDS
    for (int i = 0; sentences[i] != NULL; i++) {
        char* sentence = strdup(sentences[i]);
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

int main() {
    // EXAMPLE SENTENCES
    char* sentences[] = {
        "a quick brown fox jumps right over the lazy dog",
        "a quick brown fox jumps right over the lazy dog",
        NULL
    };

    // EXTRACT UNIQUE WORDS FROM THE SENTENCES
    extractUniqueWords(sentences);

    // PRINT THE UNIQUE WORDS
    printUniqueWords();

    return 0;
}
