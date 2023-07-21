#include <stdio.h>
#include <stdlib.h>

typedef struct _bit_stream {
    char*  buf;
    size_t len;
    size_t cap;
    size_t bit;
} bit_stream;

void print_loss(FILE* file, uint8_t po2, bit_stream* bs, float scale);
void  encode_rice(size_t po2, int delta, bit_stream* bs);
void read_loss(FILE* file, float* loss, float* scale);
void  bit_stream_set(bit_stream* bs, int set);
void bit_stream_init(bit_stream* bs);