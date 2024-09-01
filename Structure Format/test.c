#include <stdio.h>    // FOR FILE OPERATIONS AND STANDARD INPUT/OUTPUT
#include <stdlib.h>   // FOR MEMORY MANAGEMENT AND PROCESS CONTROL
#include <string.h>   // FOR STRING HANDLING

// DEFINE THE FUNCTION BEFORE MAIN
void read_weights(const char* path, double* semi_final_layer_weights, int num_weights) {
    char filename[256];
    FILE *file;
    double weight;

    // ITERATE FROM 1 TO THE SPECIFIED NUMBER OF WEIGHTS
    for (int i = 1; i <= num_weights; i++) {
        // CREATE THE FILE NAME
        snprintf(filename, sizeof(filename), "%sweight_%d.txt", path, i);

        // OPEN THE FILE FOR READING
        file = fopen(filename, "r");
        if (file == NULL) {
            fprintf(stderr, "Error opening file %s\n", filename);
            exit(EXIT_FAILURE);
        }

        // READ THE DOUBLE VALUE FROM THE FILE
        if (fscanf(file, "%lf", &weight) != 1) {
            fprintf(stderr, "Error reading value from file %s\n", filename);
            fclose(file);
            exit(EXIT_FAILURE);
        }

        // STORE THE VALUE IN THE ARRAY
        semi_final_layer_weights[i - 1] = weight;

        // CLOSE THE FILE
        fclose(file);
    }
}

int main() {
    double semi_final_layer_weights[512] = {0.0}; // DECLARE THE ARRAY

    // DEFINE THE PATH TO YOUR FILES
    const char* path = "/home/kartikey-bartwal/Technical Stuffs/C-Transformers-Unleashing-the-BERT-Beast/Structure Format/Model Trained Weights/Semi_Final Multi-layered perceptron Weights/";

    // READ WEIGHTS FROM FILES
    read_weights(path, semi_final_layer_weights, 512);

    return 0;
}
