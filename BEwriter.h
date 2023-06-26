#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _bin_arr{
    char* bf;
    int position;
    int size;
} bin_arr;

void init_bin_arr(bin_arr* ba);
void set_bin_arr(bin_arr* ba, int not_zero);
void fill_bin_arr(int rem_len, int po2, int delta, bin_arr* ba);
void BEwriter(FILE* file, double m, int po2, int* arr);