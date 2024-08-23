#include <stdio.h>
#include <unistd.h>
#include <omp.h>


int main() {

/////////////////////////////   LEVEL1: TRAINING DATA PREPARATION //////////////////////////
   
    // LOAD OUR RAW TEXT DATA 

    printf( "Loaded our raw text data \n" );



    printf( "Here is a little sample of our massive text data: \n\n\n\n\n\n" );


    sleep( 2 );

    // BASIC DATA CLEANING

    printf( " Completed with basic data cleaning \n" );


    printf( "Here is a little sample of how our text data looks after basic data cleaning \n\n\n\n\n\n");


    sleep( 2 );

    // SPLIT INTO SENTENCES



    printf( " Splitted every single text into sentences \n" );


    printf( "Here are a few sentences: \n\n\n\n\n\n"  );


    sleep( 2 );

    // WORD MAPPING DICTIONARY 


    printf( " Prepared the word mappings \n" );


    printf( " Here are some of the word mappings: \n\n\n\n\n\n" );


    sleep( 2 );


    // PREPARE BATCHES OF SAMPLES FOR TRAINING 


    printf( " Data preparation completed \n\n\n" );


    printf( " Added padding \n\n");



    sleep( 2 );

    

/////////////////////////////   LEVEL2: FEATURE ENGINEERING  //////////////////////////
   


     // REPLACE EVERY WORD WITH THEIR TOKEN MAPPING


    printf( "Token mapping completed \n\n" );

    sleep( 2 );



////////   LEVEL3: MODEL TRAINING BEGINS   //////////////////////////

#define BATCH_SIZE 32

    int epochs = 100;
    int num_samples = 1000;  // Assume we have 1000 samples, adjust as needed

    #pragma omp parallel
    {
        #pragma omp for
        for (int epoch = 1; epoch <= epochs; epoch++) {
            printf("Epoch Number: %d\n", epoch);

            float total_loss = 0.0f;

            // Process samples in batches
            for (int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
                int batch_end = batch_start + BATCH_SIZE;
                if (batch_end > num_samples) batch_end = num_samples;

                #pragma omp parallel for reduction(+:total_loss)
                for (int sample = batch_start; sample < batch_end; sample++) {
                    printf("Processing Sample: %d\n", sample);

                    // REPLACE EVERY WORD WITH THEIR WORD EMBEDDING
                    printf("Sample after replacing word by their word embedding:\n\n");

                    // ADD POSITIONAL ENCODING
                    printf("Adding positional encoding:\n\n");
                    printf("Added positional encoding\n\n");

                    // PASS THROUGH THE SELF ATTENTION LAYER
                    printf("Passing through the self attention layer\n\n");
                    printf("Attention Score Matrix:\n\n");
                    printf("Concatenated our P with the attention score matrix\n\n");

                    // FEED FORWARD LAYER
                    printf("Passing to the feed forward layer of 512 nodes\n\n");

                    // GET THE OUTPUT VECTOR
                    printf("Passing through the x-axis and y-axis node\n\n");
                    printf("Output vector embedding:\n\n");

                    // CALCULATE THE LOSS
                    float loss = 0.0f;  // Replace with actual loss calculation
                    printf("Calculating the loss\n");
                    printf("Loss: %f\n\n", loss);

                    total_loss += loss;
                }

                // Optionally, you can add a synchronization point here if needed
                #pragma omp barrier
            }

            // PERFORM BACKPROPAGATION ON THE AVERAGE LOSS
            float average_loss = total_loss / num_samples;

            // BACKPROPAGATION
            printf("Backpropagation\n\n");
            printf("Weights updated\n\n");

            // Optionally, you can add a synchronization point here if needed
            #pragma omp barrier
        }

    }


///////////////////////////////// INFERENCE /////////////////////////////////// 


    printf( " Give the input: \n\n");




    return 0;

}
