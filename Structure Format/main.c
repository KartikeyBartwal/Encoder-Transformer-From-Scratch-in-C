#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#define MAX_SENTENCE_LENGTH 512

/* Self Created header files */

#include "Data_Loading_Cleaning.h"
#include "Tokenizer.h"


int main() {

/////////////////////////////   LEVEL1: TRAINING DATA PREPARATION //////////////////////////

    // LOAD OUR RAW TEXT DATA

    char *raw_text = readFileToString("text_data.txt");  // READ THE FILE CONTENT

    printf("\n==============================\n");
    printf("       RAW TEXT DATA         \n");
    printf("==============================\n");


    for( int i = 0; i < 500; i++ ) {

        putchar( raw_text[ i ] );

    }



    // SPLIT TO SENTENCES

    printf("\n==============================\n");
    printf("    EXTRACTING SENTENCES      \n");
    printf("==============================\n");

    char **sentences = SplitSentences(raw_text);

    // PRINTING THE SENTENCES FOR CHECK

    printf("\n==============================\n");
    printf("    EXTRACTED SENTENCES       \n");
    printf("==============================\n");

    for (int i = 0; i < 5; i++ ) {
        printf("  [%d] %s\n", i + 1, sentences[i]);
    }


    // DATA CLEANING

    printf("\n==============================\n");
    printf("      CLEANED SENTENCES       \n");
    printf("==============================\n");

    for (int i = 0; i< 5; i++) {
        char *cleaned_sentence = Cleaned_Text(sentences[i]);

        free(sentences[i]); // Free the old sentence memory

        sentences[i] = cleaned_sentence; // Assign cleaned sentence back

        printf("  [%d] %s\n", i + 1, sentences[i]);
    }


    // WORD MAPPING DICTIONARY


    printf( " Prepared the word mappings \n" );


    printf( " Here are some of the word mappings: \n\n\n\n\n\n" );

    extractUniqueWords(sentences); // EXTRACT UNIQUE WORDS FROM THE SENTENCES

    Print_Tokens_And_Ids();

    sleep( 2 );


    // PREPARE BATCHES OF SAMPLES FOR TRAINING

    /*
    for every sentence in 'sentences':

        1) Convert that sentnce into an array of strings
        2) Convert that sentence into an array of integers by replacing the word by their numerical mapping
        3) If the number of elements in the array is greater than 512, trim to make it 512.
        4) If the number of elements in the array is lesser than 512, do padding by adding 0 values to the array

        Append the array into a multidimensional array, called: training_data
    */

    printf( " Preparing Training Data... \n");

    int num_sentences = 0;

    while( sentences[ num_sentences ] != NULL ) {

        num_sentences++;

    }

    int** training_data = malloc( num_sentences * sizeof( int *) );

    int training_data_count = 0;

    for( int i = 0; i < num_sentences; i++ ) {

        char* sentence = sentences[ i ];

        // 1) CONVERT SENTENCE INTO AN ARRAY OF STRINGS

        char* words[ MAX_SENTENCE_LENGTH ];

        int word_count = 0;

        char* word = strtok( sentence , " ");

        while( word != NULL && word_count < MAX_SENTENCE_LENGTH ) {

            words[ word_count++ ] = word;

            word = strtok( NULL , " ");
        }

        // 2) CONVERT SENTENCE INTO AN ARRAY OF INTEGERS

        int* token_array = malloc( MAX_SENTENCE_LENGTH * sizeof( int ) );

        for( int j = 0; j < word_count; j++ ) {

            token_array[ j ] = getTokenId( words[ j ] );

           // token_array[ j ] = 10;

        }


        // 3) Trim or pad the array to exactly 512 elements

       if( word_count > MAX_SENTENCE_LENGTH ) {

           word_count = MAX_SENTENCE_LENGTH;
       }

       else if( word_count < MAX_SENTENCE_LENGTH ) {

           for( int j = word_count; j < MAX_SENTENCE_LENGTH; j++ ) {

               token_array[ j ] = 0;
           }
       }


       // APPEND THE ARRAY TO TRAINING DATA

      training_data[ training_data_count++] = token_array;

    }

    printf( " Training data prepared. Total samples: %d\n" , training_data_count);


    // PRINT A FEW SAMPLES OF THE TRAINING DATA

   printf( " Sample of training data ( first 10 tokens of first 5 samples: \n" );

  for( int i = 0; i < 20 && i < training_data_count; i++ ) {

      printf( " Sample %d: " , i + 1);

      for( int j = 0; j < 10; j++ ) {

          printf( " %d " , training_data[ i ][ j ]);
      }

      printf( " ...\n");
  }

  printf( " Data Preparation completed\n");

  printf( " Added padding\n");

sleep( 2 );



/////////////////////////////   LEVEL2: FEATURE ENGINEERING  //////////////////////////



     // REPLACE EVERY WORD WITH THEIR TOKEN MAPPING


    printf( "Token mapping completed \n\n" );

    sleep( 2 );



////////   LEVEL3: MODEL TRAINING BEGINS   //////////////////////////

    int epochs = 1;
    int num_samples = training_data_count;  // Assume we have 1000 samples, adjust as needed


    printf("num samples: %d \n" , training_data_count );

    // RUN IT FOR 'epochs' NUMBER OF EPOCHS

   for( int epoch = 1; epoch < epochs; epoch++ ) {

       printf( " Epoch: %d \n" , epoch );


       // CURRENTLY, KEEP BATCH SIZE AS 1

       for (int i = 0; i < num_samples; i++) {

           // PRINT THE CURRENT SAMPLE NUMBER
           printf("Sample %d: ", i + 1);

           // ITERATE AND PRINT THE ELEMENTS OF THE i-th ARRAY IN TRAINING_DATA
           for (int j = 0; j < MAX_SENTENCE_LENGTH; j++) {

               printf("%d ", training_data[i][j]);
           }

           // MOVE TO THE NEXT LINE AFTER PRINTING THE ARRAY
           printf("\n");
       }

    }


    return 0;

}
