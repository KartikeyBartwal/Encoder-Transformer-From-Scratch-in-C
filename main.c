#include <stdio.h>

#include <stdlib.h> // FOR MEMORY ALLOCATION

#include <string.h> // FOR STRING HANDLING FUNCTIONS

#include <ctype.h>  // FOR CHARACTER HANDLING FUNCTIONS (E.G. , TOLOWER )


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

    int len = strlen( raw_text );

    
    // CREATE A BUFFER FOR THE CLEANED TEXT

    static char cleaned_text[ 1024 ]; // MAX LENGTH OF IS 1024

    int j = 0;


    // LOOP THROUGH EACH CHARACTER OF THE INPUT STRING

    for( int i = 0; i < len ; i++ ) {

        // CONVERT TO LOWERCASE 

        char ch = tolower( raw_text[i] );


        // REMOVE SPECIFIED CHARACTERS 

        if( ch != '.' && ch != ',' && ch != '!' && ch != '?' && ch != '\\' && ch != '-' && ch != ';' && ch != '(' && ch != ')' && ch != '-') {

            cleaned_text[ j++ ] = ch; // ADD TO CLEANED TEXT 
        }
    }

    // NULL-TERMINATE THE CLEANED TEXT 

    cleaned_text[ j ] = '\0';

    
    // RETURN THE CLEANED TEXT 

    return cleaned_text;
}



int main() {

/////////////////////////// THE DATASET /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    char *raw_text = readFileToString("text_data.txt");  // READ THE FILE CONTENT

    printf("Raw Text Data :\n%s\n", raw_text);  // PRINT THE FILE CONTENT


//////////////////////////// SPLIT TO SENTENCES //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    printf("\n Extracting Sentences: \n");

    char **sentences = SplitSentences( raw_text );

    
    // PRINTING THE SENTENCES FOR CHECK 

    printf("\n Extracted Successfully: \n");

    for( int i = 0; sentences[i] != NULL; i++ ) {

        printf(" %d: %s\n" , i + 1 , sentences[i]) ;
    }

/////////////////////////// DATA CLEANING //////////////////////////////////////////////////////////////////////////////

    printf(" Cleaned text sentences: \n" );

    for( int i = 0; sentences[i] != NULL; i++ ) {

        sentences[i] = Cleaned_Text( sentences[i] ); 

        printf(" %d: %s\n" , i + 1 , sentences[i]) ;

    }


/////////////////////// TOKENIZATION AND NUMERICALIZATION ////////////////











////////////////////////////// FREE THE MEMORY AFTER RUNNING THE ENTIRE PROGRAM AND THE MODEL //////////////////////////



    free( raw_text );  // FREE THE ALLOCATED MEMORY

    return 0;  // EXIT THE PROGRAM SUCCESSFULLY

}