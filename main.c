#include <stdio.h>

#include <stdlib.h> // FOR MEMORY ALLOCATION

#include <string.h> // FOR STRING HANDLING FUNCTIONS

#include <ctype.h>  // FOR CHARACTER HANDLING FUNCTIONS (E.G. , TOLOWER )

/* Self Created header files */

#include "Data_Loading_Cleaning.h"
#include "Tokenizer.h"


int main() {

    // THE DATASET

    char *raw_text = readFileToString("text_data.txt");  // READ THE FILE CONTENT

    printf("\n==============================\n");
    printf("       RAW TEXT DATA         \n");
    printf("==============================\n");
    printf("%s\n", raw_text);  // PRINT THE FILE CONTENT


    // SPLIT TO SENTENCES

    printf("\n==============================\n");
    printf("    EXTRACTING SENTENCES      \n");
    printf("==============================\n");

    char **sentences = SplitSentences(raw_text);

    // PRINTING THE SENTENCES FOR CHECK

    printf("\n==============================\n");
    printf("    EXTRACTED SENTENCES       \n");
    printf("==============================\n");

    for (int i = 0; sentences[i] != NULL; i++) {
        printf("  [%d] %s\n", i + 1, sentences[i]);
    }


    // DATA CLEANING

    printf("\n==============================\n");
    printf("      CLEANED SENTENCES       \n");
    printf("==============================\n");

    for (int i = 0; sentences[i] != NULL; i++) {
        char *cleaned_sentence = Cleaned_Text(sentences[i]); 

        free(sentences[i]); // Free the old sentence memory

        sentences[i] = cleaned_sentence; // Assign cleaned sentence back

        printf("  [%d] %s\n", i + 1, sentences[i]);
    }


    // TOKENIZATION AND NUMERICALIZATION

    printf("\n==============================\n");
    printf("    TOKENIZATION AND NUMERICALIZATION \n");
    printf("==============================\n");

    extractUniqueWords(sentences); // EXTRACT UNIQUE WORDS FROM THE SENTENCES

    Print_Tokens_And_Ids();


    // 

    ////////////////////////////// FREE THE MEMORY AFTER RUNNING THE ENTIRE PROGRAM AND THE MODEL //////////////////////////

    free(raw_text);  // FREE THE ALLOCATED MEMORY

    // FREE SENTENCES MEMORY
    for (int i = 0; sentences[i] != NULL; i++) {
        free(sentences[i]);
    }
    free(sentences); // Free the array of sentences

    printf("\n==============================\n");
    printf("          PROGRAM ENDED       \n");
    printf("==============================\n");

    return 0;  // EXIT THE PROGRAM SUCCESSFULLY
}