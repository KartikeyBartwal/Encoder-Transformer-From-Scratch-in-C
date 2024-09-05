#include <math.h>
#include <stdio.h>

// FUNCTION TO CALCULATE MEAN SQUARED ERROR (MSE)
double calculate_mse(double output_array[], double expected_output_array[], int size) {
    double mse = 0.0;
    for (int i = 0; i < size; i++) {
        double difference = output_array[i] - expected_output_array[i];
        mse += difference * difference;
    }
    return ( mse / size );

}

double clip_gradient_backpropagation(double gradient, double clip_threshold) {
    if (gradient > clip_threshold) {
        return clip_threshold;
    } else if (gradient < -clip_threshold) {
        return -clip_threshold;
    }
    return gradient;
}

// void update_weights_last_layer(double loss, double learning_rate , double final_layer_weights[], double semi_final_layer_weights[], int final_layer_size, int semi_final_layer_size) {

//     // ITERATE THROUGH EACH WEIGHT FOR NODE 1
//     for (int i = 0; i < semi_final_layer_size; i++) {

//         // CALCULATE THE GRADIENT FOR THE WEIGHT OF NODE 1
//         double gradient_node_1 = semi_final_layer_weights[i] * loss;

//         // UPDATE WEIGHT OF NODE 1
//         final_layer_weights[i] -= learning_rate * gradient_node_1;
//     }

//     // ITERATE THROUGH EACH WEIGHT FOR NODE 2
//     for (int i = 0; i < semi_final_layer_size; i++) {

//         // CALCULATE THE GRADIENT FOR THE WEIGHT OF NODE 2
//         double gradient_node_2 = semi_final_layer_weights[i] * loss;

//         // UPDATE WEIGHT OF NODE 2
//         final_layer_weights[i + semi_final_layer_size] -= learning_rate * gradient_node_2;
//     }

//     // PRINT UPDATED WEIGHTS FOR VERIFICATION (OPTIONAL)
//     printf("Updated weights for final layer:\n");
//     for (int i = 0; i < 10; i++) {
//         printf("final_layer_weights[%d] = %f\n", i, final_layer_weights[i]);
//     }
//     printf(".\n.\n.\n");
//     printf("final_layer_weights[%d] = %f\n", final_layer_size - 2, final_layer_weights[final_layer_size - 2]);
//     printf("final_layer_weights[%d] = %f\n", final_layer_size - 1, final_layer_weights[final_layer_size - 1]);
// }

void update_weights_last_layer(double loss, double learning_rate, double final_layer_weights[], double semi_final_layer_weights[], int final_layer_size, int semi_final_layer_size, double clip_threshold) {

    // ITERATE THROUGH EACH WEIGHT FOR NODE 1
    for (int i = 0; i < semi_final_layer_size; i++) {

        // CALCULATE THE GRADIENT FOR THE WEIGHT OF NODE 1
        double gradient_node_1 = semi_final_layer_weights[i] * loss;

        // CLIP THE GRADIENT
        gradient_node_1 = clip_gradient_backpropagation(gradient_node_1, clip_threshold);

        // UPDATE WEIGHT OF NODE 1
        final_layer_weights[i] -= learning_rate * gradient_node_1;
    }

    // ITERATE THROUGH EACH WEIGHT FOR NODE 2
    for (int i = 0; i < semi_final_layer_size; i++) {

        // CALCULATE THE GRADIENT FOR THE WEIGHT OF NODE 2
        double gradient_node_2 = semi_final_layer_weights[i] * loss;

        // CLIP THE GRADIENT
        gradient_node_2 = clip_gradient_backpropagation(gradient_node_2, clip_threshold);

        // UPDATE WEIGHT OF NODE 2
        final_layer_weights[i + semi_final_layer_size] -= learning_rate * gradient_node_2;
    }

    // PRINT UPDATED WEIGHTS FOR VERIFICATION (OPTIONAL)
    printf("Updated weights for final layer:\n");
    for (int i = 0; i < 10; i++) {
        printf("final_layer_weights[%d] = %f\n", i, final_layer_weights[i]);
    }
    printf(".\n.\n.\n");
    printf("final_layer_weights[%d] = %f\n", final_layer_size - 2, final_layer_weights[final_layer_size - 2]);
    printf("final_layer_weights[%d] = %f\n", final_layer_size - 1, final_layer_weights[final_layer_size - 1]);
}

void update_semi_final_layer_weights(double loss, double learning_rate, double semi_final_layer_weights[], int semi_final_layer_size, double clip_threshold) {

    // ITERATE THROUGH EACH WEIGHT IN THE SEMI FINAL LAYER
    for (int i = 0; i < semi_final_layer_size; i++) {

        // CALCULATE THE GRADIENT FOR THE SEMI FINAL LAYER WEIGHT
        double gradient = loss * semi_final_layer_weights[i];

        // CLIP THE GRADIENT
        gradient = clip_gradient_backpropagation(gradient, clip_threshold);

        // UPDATE SEMI FINAL LAYER WEIGHT
        semi_final_layer_weights[i] -= learning_rate * gradient;
    }

    // PRINT UPDATED SEMI FINAL LAYER WEIGHTS FOR VERIFICATION (OPTIONAL)
    printf("Updated weights for semi-final layer:\n");
    for (int i = 0; i < 10; i++) {
        printf("semi_final_layer_weights[%d] = %f\n", i, semi_final_layer_weights[i]);
    }
    printf(".\n.\n.\n");
    printf("semi_final_layer_weights[%d] = %f\n", semi_final_layer_size - 2, semi_final_layer_weights[semi_final_layer_size - 2]);
    printf("semi_final_layer_weights[%d] = %f\n", semi_final_layer_size - 1, semi_final_layer_weights[semi_final_layer_size - 1]);

}

// void update_attention_matrices(double loss, double LEARNING_RATE, double k_matrix[ ][ ], double q_matrix[ ][ ], double v_matrix[][ ] , int MATRIX_SIZE) {

//     // UPDATE K MATRIX
//     printf("Updating K Matrix:\n");
//     for (int i = 0; i < MATRIX_SIZE; i++) {
//         for (int j = 0; j < MATRIX_SIZE; j++) {
//             // Example gradient calculation; adjust as needed
//             double gradient = loss * k_matrix[i][j];
//             k_matrix[i][j] -= LEARNING_RATE * gradient;

//             // PRINT UPDATED ELEMENT FOR VERIFICATION
//             printf("k_matrix[%d][%d] = %lf\n", i, j, k_matrix[i][j]);
//         }
//     }
//     printf("\n");

//     // UPDATE Q MATRIX
//     printf("Updating Q Matrix:\n");
//     for (int i = 0; i < MATRIX_SIZE; i++) {
//         for (int j = 0; j < MATRIX_SIZE; j++) {
//             // Example gradient calculation; adjust as needed
//             double gradient = loss * q_matrix[i][j];
//             q_matrix[i][j] -= LEARNING_RATE * gradient;

//             // PRINT UPDATED ELEMENT FOR VERIFICATION
//             printf("q_matrix[%d][%d] = %lf\n", i, j, q_matrix[i][j]);
//         }
//     }
//     printf("\n");

//     // UPDATE V MATRIX
//     printf("Updating V Matrix:\n");
//     for (int i = 0; i < MATRIX_SIZE; i++) {
//         for (int j = 0; j < MATRIX_SIZE; j++) {
//             // Example gradient calculation; adjust as needed
//             double gradient = loss * v_matrix[i][j];
//             v_matrix[i][j] -= LEARNING_RATE * gradient;

//             // PRINT UPDATED ELEMENT FOR VERIFICATION
//             printf("v_matrix[%d][%d] = %lf\n", i, j, v_matrix[i][j]);
//         }
//     }
//     printf("\n");
// }
