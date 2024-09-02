#include <math.h>

// FUNCTION TO CALCULATE MEAN SQUARED ERROR (MSE)
double calculate_mse(double output_array[], double expected_output_array[], int size) {
    double mse = 0.0;
    for (int i = 0; i < size; i++) {
        double difference = output_array[i] - expected_output_array[i];
        mse += difference * difference;
    }
    return ( mse / size );

}
