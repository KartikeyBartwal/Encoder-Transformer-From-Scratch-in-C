#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <math.h>
#include <stdlib.h>

///////////////// POSITIONAL ENCODING /////////////////////////

double* positional_encoding( int index, int vector_size );


////////////////////  SELF ATTENTION //////////////////////////////////

#define MATRIX_SIZE 2

// Declare matrices
extern double k_matrix[MATRIX_SIZE][MATRIX_SIZE];
extern double q_matrix[MATRIX_SIZE][MATRIX_SIZE];
extern double v_matrix[MATRIX_SIZE][MATRIX_SIZE];

// Function to read a matrix from a file
double read_single_value_from_file(const char* filename);

// Function to initialize matrices with weights from files
void initialize_matrices_from_files();

// Function to print a matrix
void print_matrix(const char* name, double matrix[MATRIX_SIZE][MATRIX_SIZE]);


#endif
