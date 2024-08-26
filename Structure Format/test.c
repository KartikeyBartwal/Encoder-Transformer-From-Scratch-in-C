#include <stdio.h>
#include "transformer_block.h"
#include <math.h>
#include <stdlib.h>


int main() {

    initialize_matrices_from_files();

    print_matrix( "Key Matrix" , k_matrix);

    print_matrix( "Query Matrix" , q_matrix );

    print_matrix( "Value Matrix" , v_matrix );

    // // DEFINE POSITION INDEX AND VECTOR SIZE
    // int index = 5;
    // int vector_size = 10;

    // // CALL THE POSITIONAL ENCODING FUNCTION
    // double* encoding = positional_encoding(index, vector_size);

    // // PRINT THE ENCODING VALUES
    // for (int i = 0; i < vector_size; i++) {
    //     printf("%f ", encoding[i]);
    // }

    // // FREE THE ALLOCATED MEMORY
    // free(encoding);

    return 0;
}
