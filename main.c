#include <stdio.h>

#include <stdlib.h> // FOR MEMORY ALLOCATION

#include <string.h> // FOR STRING HANDLING FUNCTIONS

#include <ctype.h>  // FOR CHARACTER HANDLING FUNCTIONS (E.G. , TOLOWER )

#define TABLE_SIZE 5000  // SIZE OF THE HASH TABLE



// GLOBAL TOKEN ID STARTING FROM 4
int global_token = 4;

// FUNCTION TO READ THE ENTIRE CONTENT OF A FILE AND RETURN IT AS A STRING
char* readFileToString(const char *filename) {

    FILE *filePointer = fopen(filename, "r");  // OPEN THE FILE IN READ MODE

    fseek(filePointer, 0, SEEK_END);  // MOVE FILE POINTER TO THE END OF THE FILE

    long fileSize = ftell(filePointer);  // GET THE SIZE OF THE FILE

    rewind(filePointer);  // RESET FILE POINTER TO THE BEGINNING OF THE FILE

    char *fileContent = (char *)malloc(fileSize + 1);  // ALLOCATE MEMORY FOR THE FILE CONTENT

    fread(fileContent, 1, fileSize, filePointer);  // READ THE ENTIRE FILE INTO THE STRING

    fileContent[fileSize] = '\0';  // NULL TERMINATE THE STRING

    fclose(filePointer);  // CLOSE THE FILE

    return fileContent;  // RETURN THE FILE CONTENT AS A STRING
}

char** SplitSentences( char *raw_text ) {

    // PUNCTUATION MARK USED TO SPLIT SENTENCES 

    const char *delimiters = ".!?";

    // ALLOCATE MEMORY FOR A MAXIMUM OF 100 SENTENCES

    char **sentences = malloc( 100 * sizeof( char* ));


    // SPLIT THE TEXT INTO SENTENCES 

    char *token = strtok( raw_text , delimiters );

    int index = 0;

    // EXTRACT SENTENCES

    while( token != NULL ) {

        // ALLOCATE MEMORY FOR EACH SENTENCE

        sentences[ index ] = malloc( ( strlen( token ) + 1 ) * sizeof( char ));

        strcpy( sentences[ index ] , token );

        index++;

        token = strtok( NULL , delimiters);

    }   

    // NULL TERMINATE THE LAST ELEMENT 

    sentences[ index ] = NULL;

    return sentences;  // RETURN THE ARRAY OF SENTENCES 


};


char* Cleaned_Text(char *raw_text) {

    // GET THE LENGTH OF THE INPUT STRING

    int len = strlen(raw_text);


    // ALLOCATE MEMORY FOR THE CLEANED TEXT DYNAMICALLY

    char *cleaned_text = (char *)malloc((len + 1) * sizeof(char)); // +1 FOR NULL TERMINATION


    // CHECK FOR MEMORY ALLOCATION FAILURE

    if (!cleaned_text) {

        fprintf(stderr, "Memory allocation failed for cleaned text\n");

        exit(EXIT_FAILURE);
    }


    int j = 0;

    // FLAG TO INDICATE IF THE LAST CHARACTER WAS A SPACE

    int last_was_space = 0;


    // LOOP THROUGH EACH CHARACTER OF THE INPUT STRING

    for (int i = 0; i < len; i++) {

        // CONVERT TO LOWERCASE 

        char ch = tolower(raw_text[i]);


        // REMOVE SPECIFIED CHARACTERS 

        if (isalnum(ch)) { // KEEP ONLY ALPHANUMERIC CHARACTERS

            cleaned_text[j++] = ch; // ADD TO CLEANED TEXT

            last_was_space = 0; // RESET THE SPACE FLAG
        } else if (ch == ' ' && !last_was_space) {

            cleaned_text[j++] = ' '; // ADD SPACE ONLY IF LAST CHAR WAS NOT SPACE

            last_was_space = 1; // SET THE SPACE FLAG
        }
    }


    // NULL-TERMINATE THE CLEANED TEXT 

    cleaned_text[j] = '\0';


    // RETURN THE CLEANED TEXT 

    return cleaned_text;
}




// STRUCTURE FOR THE HASH TABLE NODE
typedef struct word_token_node { // CHANGED NODE TO WORD_TOKEN_NODE

    char *word;              // THE WORD

    unsigned int token_id;   // THE UNIQUE TOKEN ID

    struct word_token_node *next; // POINTER TO THE NEXT NODE IN THE LINKED LIST

} word_token_node;


// HASH TABLE TO STORE UNIQUE WORDS
word_token_node* hashTable[TABLE_SIZE] = { NULL };


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