#ifndef DATA_LOADING_CLEANING_H
#define DATA_LOADING_CLEANING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// FUNCTION TO READ THE ENTIRE CONTENT OF A FILE AND RETURN IT AS A STRING
char* readFileToString(const char *filename);

// FUNCTION TO SPLIT TEXT INTO SENTENCES BASED ON SPECIFIC DELIMITERS
char** SplitSentences(char *raw_text);

// FUNCTION TO CLEAN TEXT BY CONVERTING TO LOWERCASE AND REMOVING NON-ALPHANUMERIC CHARACTERS
char* Cleaned_Text(char *raw_text);

#endif // Data_Loading_Cleaning.h
