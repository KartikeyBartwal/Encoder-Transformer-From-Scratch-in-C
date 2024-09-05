#ifndef DATA_PREPROCESSING_H

#define DATA_PREPROCESSING_H


#include <stddef.h>  // FOR SIZE_T
#include <float.h> // For FLT_MAX
#define MATRIX_SIZE 2
#define EMBEDDING_SIZE 512

// FUNCTION TO STANDARDIZE A 2D ARRAY GIVEN AS A LINEARIZED FLOAT ARRAY
// `data` IS A LINEARIZED 2D ARRAY WITH DIMENSIONS `ROWS` X `COLS`
// THE FUNCTION WILL MODIFY THE `DATA` IN-PLACE

void min_max_normalize_and_scale(float* data, size_t size, float new_min, float new_max);
size_t get_meaningful_length(const float* data, size_t size);
void Add_Positional_Encoding(float embedding_matrix[][2], int max_sentence_length);
void scale_matrix(float matrix[ EMBEDDING_SIZE][MATRIX_SIZE]);

#endif // DATA_PREPROCESSING_H
