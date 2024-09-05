#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include <omp.h>

#define MAX_SENTENCE_LENGTH 512
#define MATRIX_SIZE 2
#define EMBEDDING_DIM 2
#define LEARNING_RATE 0.01

/* SELF CREATED HEADER FILES */

#include "Data_Loading_Cleaning.h"

#include "Tokenizer.h"

#include "Data_Preprocessing.h"

#include "transformer_block.h"

#include "feed_forward_layer.h"

#include "activation_functions.h"

#include "backpropagation.h"

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


        free(sentences[i]); // FREE THE OLD SENTENCE MEMORY

        sentences[i] = cleaned_sentence; // ASSIGN CLEANED SENTENCE BACK


        printf("  [%d] %s\n", i + 1, sentences[i]);

    }


    // WORD MAPPING DICTIONARY

    printf( " PREPARED THE WORD MAPPINGS \n" );


    sleep( 2 );


    printf( " GENERATING WORD MAPPINGS FOR EVERY WORD ");


    printf( " HERE ARE SOME OF THE WORD MAPPINGS: \n\n\n\n\n\n" );


    extractUniqueWords(sentences); // EXTRACT UNIQUE WORDS FROM THE SENTENCES

    Print_Tokens_And_Ids();


    sleep( 2 );


    // PREPARE BATCHES OF SAMPLES FOR TRAINING

    /*
    for every sentence in 'sentences':

        1) Convert that sentence into an array of strings
        2) Convert that sentence into an array of integers by replacing the word by their numerical mapping
        3) If the number of elements in the array is greater than 512, trim to make it 512.
        4) If the number of elements in the array is lesser than 512, do padding by adding 0 values to the array

        Append the array into a multidimensional array, called: training_data
    */


    printf( " PREPARING TRAINING DATA... \n");


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


        // 3) TRIM OR PAD THE ARRAY TO EXACTLY 512 ELEMENTS

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


    printf( " TRAINING DATA PREPARED. TOTAL SAMPLES: %d\n" , training_data_count);


    // PRINT A FEW SAMPLES OF THE TRAINING DATA

    printf( " SAMPLE OF TRAINING DATA ( FIRST 10 TOKENS OF FIRST 5 SAMPLES: \n" );


    for( int i = 0; i < 20 && i < training_data_count; i++ ) {

        printf( " SAMPLE %d: " , i + 1);


        for( int j = 0; j < 10; j++ ) {

            printf( " %d " , training_data[ i ][ j ]);

        }


        printf( " ...\n");

    }

    printf( " DATA PREPARATION COMPLETED\n");

    printf( " ADDED PADDING\n");

    sleep( 2 );


/////////////////////////////   LEVEL2: FEATURE ENGINEERING  //////////////////////////


    // REPLACE EVERY WORD WITH THEIR TOKEN MAPPING

    printf( "TOKEN MAPPING COMPLETED \n\n" );

    sleep( 2 );


//////////////////// GENERATE WORD EMBEDDINGS FOR EVERY SINGLE WORD PRESENT IN THE VOCABULARY


////////   LEVEL3: MODEL TRAINING BEGINS   //////////////////////////

int epochs = 100;

int num_samples = training_data_count;  // ASSUME WE HAVE 1000 SAMPLES, ADJUST AS NEEDED

printf("NUM SAMPLES: %d \n", training_data_count);




// LOAD EVERYTHING YOU NEED TO LOAD INTO RAM BEFORE YOU START TRAINING THE MODEL


////////////////////////// LOAD THE SELF ATTENTION BLOCK /////////////////////////////////
initialize_matrices_from_files();

///////////////////////// SEMI FINAL LAYER NODES//////////////////////////////////
double semi_final_layer_weights[ 512 * 65] = {0.0};
const char* path = "/home/kartikey-bartwal/Technical Stuffs/C-Transformers- MAIN BRANCH /Structure Format/Model Trained Weights/Semi_Final Multi-layered perceptron Weights/";
read_weights(path, semi_final_layer_weights, 512 * 64 );


///////////////////////// FINAL LAYER NODES//////////////////////////////////
double final_layer_weights[ 65* 2] = { 0.0 };

const char* path_2 = "/home/kartikey-bartwal/Technical Stuffs/C-Transformers- MAIN BRANCH /Structure Format/Model Trained Weights/Final Multi-layered perceptron weights/";
read_weights( path_2 , final_layer_weights , 64 * 2);


// EVERY EPOCH
for (int epoch = 0; epoch < epochs; epoch++) {

    printf("Epoch: %d\n", epoch);

    double total_loss = 0;

    // CURRENTLY, KEEPING A BATCH SIZE OF 1. HENCE PROCESSING EVERY SAMPLE ONE BY ONE

    for (int sample_index = 0; sample_index < num_samples; sample_index++) {

        printf("Sample %d: ", sample_index + 1);

        float* sentence = (float*)malloc(MAX_SENTENCE_LENGTH * sizeof(float));

        if (sentence == NULL) {

            printf("Memory allocation failed.\n");

            return 1;

        }

        for (int sentence_index = 0; sentence_index < MAX_SENTENCE_LENGTH; sentence_index++) {

            sentence[sentence_index] = training_data[sample_index][sentence_index];

        }

        printf("Current sample's first 10 elements: ");

        for (int sentence_index = 0; sentence_index < 10; sentence_index++) {

            printf(" %f, ", sentence[sentence_index]);

        }

        printf("\n");

        int y_actual = 0;

        for (int k = MAX_SENTENCE_LENGTH - 1; k >= 0; k--) {

            if (sentence[k] != 0) {

                y_actual = sentence[k];

                sentence[k] = 0;

                break;

            }

        }

        printf("y_actual token: %d \n" , y_actual);

        printf("max sentence length: %d \n", MAX_SENTENCE_LENGTH);
        float embedding_matrix[MAX_SENTENCE_LENGTH][2] = {0}; // 512 x 2 MATRIX

        for (int i = 0; i < MAX_SENTENCE_LENGTH; i++) {

            if (sentence[i] != 0) {

                unsigned int token_id = (unsigned int)sentence[i];

                double curr_vector_embedding[ 2 ];

                getEmbeddingByTokenId( token_id , curr_vector_embedding );

                embedding_matrix[i][0] = ( float ) curr_vector_embedding[0];

                embedding_matrix[i][1] = ( float ) curr_vector_embedding[1];


            } else {

                embedding_matrix[i][0] = 0.0f;

                embedding_matrix[i][1] = 0.0f;

            }

        }

        sleep(2);

        printf("Embedding Matrix:\n");

        for (int i = 0; i < 10; i++) {

            printf(" [%f, %f] \n", embedding_matrix[i][0], embedding_matrix[i][1]);

            if ((i + 1) % 10 == 0) {

                break;

            }

        }

        printf(" Scaling down the matrix values: \n\n");

        scale_matrix( embedding_matrix );

        printf(" After scaling: \n");

        printf("Embedding Matrix:\n");

        for (int i = 0; i < 10; i++) {

            printf(" [%f, %f] \n", embedding_matrix[i][0], embedding_matrix[i][1]);

            if ((i + 1) % 10 == 0) {

                break;

            }
        }

        printf("Processing Sentence %d...\n", sample_index + 1);

        sleep(2);

        printf("Adding positional encoding values to the matrix \n");

        Add_Positional_Encoding(embedding_matrix, MAX_SENTENCE_LENGTH);

        printf("Matrix Post Positional Encoding:\n");

        for (int i = 0; i < 10; i++) {

            printf(" [%f, %f] \n", embedding_matrix[i][0], embedding_matrix[i][1]);

            if ((i + 1) % 10 == 0) {

                break;

            }

        }

        // SELF ATTENTION BLOCK

        printf(" Self attention block \n");


        // PRINTING K MATRIX
        printf("K Matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%lf ", k_matrix[i][j]); // PRINT ELEMENT OF K MATRIX
            }
            printf("\n"); // NEW LINE AFTER EACH ROW
        }

        printf("\n"); // SEPARATION BETWEEN MATRICES

        // PRINTING Q MATRIX
        printf("Q Matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%lf ", q_matrix[i][j]); // PRINT ELEMENT OF Q MATRIX
            }
            printf("\n"); // NEW LINE AFTER EACH ROW
        }

        printf("\n"); // SEPARATION BETWEEN MATRICES

        // PRINTING V MATRIX
        printf("V Matrix:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%lf ", v_matrix[i][j]); // PRINT ELEMENT OF V MATRIX
            }
            printf("\n"); // NEW LINE AFTER EACH ROW
        }

        // MULTIPLY THESE MATRICES WITH THE EMBEDDING MATRIX

        double final_k_matrix[ MAX_SENTENCE_LENGTH ][ MATRIX_SIZE] = {0};
        double final_q_matrix[ MAX_SENTENCE_LENGTH ][ MATRIX_SIZE] = {0};
        double final_v_matrix[ MAX_SENTENCE_LENGTH ][ MATRIX_SIZE] = {0};


        // MULTIPLY EMBEDDING MATRIX WITH K, Q, V MATRICES
        for (int i = 0; i < MAX_SENTENCE_LENGTH; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    final_k_matrix[i][j] += embedding_matrix[i][k] * k_matrix[k][j];
                    final_q_matrix[i][j] += embedding_matrix[i][k] * q_matrix[k][j];
                    final_v_matrix[i][j] += embedding_matrix[i][k] * v_matrix[k][j];
                }
            }
        }

        // PRINT FINAL MATRICES
        printf("Final K Matrix:\n");
        for (int i = 0; i < 10; i++) {
            printf("[%lf, %lf]\n", final_k_matrix[i][0], final_k_matrix[i][1]);
        }

        printf("Final Q Matrix:\n");
        for (int i = 0; i < 10; i++) {
            printf("[%lf, %lf]\n", final_q_matrix[i][0], final_q_matrix[i][1]);
        }

        printf("Final V Matrix:\n");
        for (int i = 0; i < 10; i++) {
            printf("[%lf, %lf]\n", final_v_matrix[i][0], final_v_matrix[i][1]);
        }


        // COMPUTE SELF-ATTENTION MATRIX SCORES
        double self_attention_matrix[ MAX_SENTENCE_LENGTH ][ MATRIX_SIZE ] = { 0 };
        compute_self_attention(embedding_matrix, final_k_matrix, final_q_matrix, final_v_matrix, MAX_SENTENCE_LENGTH, self_attention_matrix);

        // PRINT THE SELF-ATTENTION MATRIX

        printf(" \n\nSelf-Attention Matrix: \n");

        for( int i = 0; i < 10; i++ ) {

            for( int j = 0; j < MATRIX_SIZE; j++ ) {

                printf( "%lf ", self_attention_matrix[ i ][ j ] );
            }

            printf(" \n");
        }



        // ADD THE 'self_attention_matrix' AND THE 'embedding_matrix'

        double context_matrix[ MAX_SENTENCE_LENGTH ][ 2 ] = { 0 };

        add_matrices( embedding_matrix , self_attention_matrix , context_matrix, MAX_SENTENCE_LENGTH, MATRIX_SIZE );

        printf("\n\nContext Matrix (embedding_matrix + self_attention_matrix:\n");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%f ", context_matrix[i][j]);
            }
            printf("\n");
        }

        printf("\n\n");


        //////////////////////////////////// FEED FORWARD LAYER ////////////////////


        // INCREMENT THE ABSOLUTE VALUE OF THE COLUMN 1 BY 1

        for( int i = 0; i < MAX_SENTENCE_LENGTH; i++ ) {

            if( context_matrix[ i ][ 1 ] < 0 ) {
                context_matrix[ i ][ 1 ] += -100;
            }

            else{
                context_matrix[ i ][ 1 ] += 100;
            }

        }


        printf("\n\n");

        // PRINT THE FIRST 10 WEIGHTS TO VERIFY
        for (int i = 0; i < 10; i++) {
            printf("semi_final_layer_weights[%d] = %f\n", i, semi_final_layer_weights[i]);
        }
        printf(".\n.\n.\n");
        printf("semi_final_layer_weights[%d] = %f\n", 511, semi_final_layer_weights[510]);
        printf("semi_final_layer_weights[%d] = %f\n", 512, semi_final_layer_weights[511]);


        double semi_final_layer_nodes[ 512 * 65] = {0.0};

        for( int row = 0; row < 512; row++ ) {

            for( int node = 0; node < 65; node++ ) {

                int weight_index = ( row * 65 ) + node;

                double weight = semi_final_layer_weights[ weight_index ];

                double value = embedding_matrix[ row ][ 0 ] * weight + embedding_matrix[ row ][ 1 ] * weight;

                semi_final_layer_nodes[ row ] = leaky_relu( value  , 0.01);
            }
        }

        printf("\n\n Semi Final Layer Node Values: \n");

        for( int i = 0; i < 10; i++ ) {

            printf("%lf \n" , semi_final_layer_nodes[ i ]);
        }

        printf(".\n.\n.\n");

        for( int i = 500; i < 65 * 2; i++ ) {

            printf("%lf \n", semi_final_layer_nodes[ i ]);
        }


        printf("\n\n");

        for (int i = 0; i < 10; i++) {
            printf("final_layer_weights[%d] = %f\n", i, final_layer_weights[i]);
        }
        printf(".\n.\n.\n");
        printf("final_layer_weights[%d] = %f\n", 130, final_layer_weights[ 130 ]);
        printf("final_layer_weights[%d] = %f\n", 130 , final_layer_weights[ 130 ]);



        double final_node_values[ 2 ] = {0.0};

        // 0 - 512
        // 1 - 513
        // 2 - 514
        // 3 - 515
        //     .
        //     .
        //     .
        //     .
        // 511 - 1023

        // VALUE FOR NODE 1

        double total_value_node_1 = 0;

        for( int i = 0; i < 130; i++ ) {

            total_value_node_1 += semi_final_layer_nodes[ i ] * final_layer_weights[ i ];
        }

        // PASS total_value_node_1 TO AN ACTIVATION FUNCTION


        // VALUE FOR NODE 2

        double total_value_node_2 = 0;

        for( int i = 0; i < 130; i++ ) {

            total_value_node_2 += semi_final_layer_nodes[ i ] * final_layer_weights[ i + 512 ];
        }

        // HERE IS THE MODEL'S OUTPUT

        // APPLY THE SWISH ACTIVATION FUNCTION TO THE OVERALL OUTPUT
        double output_embedding[ 2 ] = { swish( total_value_node_1 ) , swish( total_value_node_2 ) };

        // printf("\n\nOutput Embedding: %lf, %lf \n", output_embedding[ 0 ], output_embedding[ 1 ] );


        ////////////////// COMPUTE THE LOSS FOR BACKPROPAGATION ////////////////////

        double expected_embedding[ 2 ];

        getEmbeddingByTokenId( y_actual, expected_embedding );

        // printf("\n Expected Embedding: %lf, %lf \n", expected_embedding[ 0 ], expected_embedding[ 1 ] );

        double loss = calculate_mse( output_embedding , expected_embedding , 2 );

        printf(" loss: %lf \n\n\n" , loss);

        total_loss += loss;

        ///////////////////////////////////////// BACKPROPAGATION ///////////////////////////////////////

        double learning_rate = ( double ) LEARNING_RATE;

        // UPDATE THE LAST LAYER WEIGHTS
        update_weights_last_layer( loss , learning_rate , final_layer_weights , semi_final_layer_weights, 1024 , 512, 100 );

        printf("\n\n");

        // UPDATE THE SECOND LAYER LAYER WEIGHTS

        update_semi_final_layer_weights(loss, learning_rate, semi_final_layer_weights, 512, 100);

        // UPDATE THE ATTENTION MATRICES

        update_attention_matrices( loss, learning_rate);


        printf("UPDATED WEIGHTS FOR THE LAST LAYER \n\n\n");
    }

    printf("********************************************** Epoch %d  total loss: %f ******************************************************************* \n\n" , epoch , total_loss);

}


    return 0;

}
