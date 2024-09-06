#include <stdio.h>

#include <stdlib.h> // FOR MEMORY ALLOCATION

#include <string.h> // FOR STRING HANDLING FUNCTIONS

#include <ctype.h>  // FOR CHARACTER HANDLING FUNCTIONS (E.G. , TOLOWER )




// FUNCTION TO READ THE ENTIRE CONTENT OF A FILE AND RETURN IT AS A STRING
char* readFileToString(const char *filename) {

    FILE *filePointer = fopen(filename, "r");  // OPEN THE FILE IN READ MODE

    fseek(filePointer, 0, SEEK_END);  // MOVE FILE POINTER TO THE END OF THE FILE

    long fileSize = ftell(filePointer);  // GET THE SIZE OF THE FILE

    rewind(filePointer);  // RESET FILE POINTER TO THE BEGINNING OF THE FILE

    char *fileContent = (char *)malloc(fileSize + 1);  // ALLOCATE MEMORY FOR THE FILE CONTENT

    fread(fileContent, 1, fileSize, filePointer);  // READ THE ENTIRE FILE INTO THE STRING

    fileContent[fileSize] = '\0';  // NULL TERMINATE THE STRING

    fclose(filePointer);  

    return fileContent;  
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

    return sentences;  


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
