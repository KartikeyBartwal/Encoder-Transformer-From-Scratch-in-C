#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <math.h>
#include <stdlib.h>

///////////////// POSITIONAL ENCODING /////////////////////////

double* positional_encoding( int index, int vector_size );


////////////////////  SELF ATTENTION //////////////////////////////////

#define MATRIX_SIZE 2
#define EMBEDDING_DIM 2

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

void dot_product(double A[MATRIX_SIZE][MATRIX_SIZE], double B[MATRIX_SIZE][MATRIX_SIZE], double result[MATRIX_SIZE][MATRIX_SIZE]);

void transpose(double matrix[MATRIX_SIZE][MATRIX_SIZE], double transposed[MATRIX_SIZE][MATRIX_SIZE]);

void calculate_attention(double Q[MATRIX_SIZE][MATRIX_SIZE], double K[MATRIX_SIZE][MATRIX_SIZE], double V[MATRIX_SIZE][MATRIX_SIZE], double result[MATRIX_SIZE][MATRIX_SIZE]);

void matrix_multiply(const float A[MATRIX_SIZE][EMBEDDING_DIM],
                     const float B[EMBEDDING_DIM][EMBEDDING_DIM],
                     float result[MATRIX_SIZE][EMBEDDING_DIM]);

#endif
