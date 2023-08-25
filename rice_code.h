#include <stdio.h>
#include <stdlib.h>

#define BUFFER_LEN 2048

typedef struct _bit_stream {
	char*  buf;
    size_t len;
    size_t cap;
    size_t bit;
} bit_stream;

/**
* print_loss
* @breif Writes into the file power of 2, that used in rice_code, and scale. After that it writes into the file loss, that located in bit_stream stucture.
* @param FILE* file - file to write
* @param short po2 - power of 2 to write
* @param bit_stream* bs - pointer to bit_stream with encoded loss
* @param float - scale to write
*/

void print_loss(FILE* file, short po2, bit_stream* bs, float scale);

/**
* encode_rice
* @breif This function encodes by rice code integer value and writes a result into bit_stream structure
* @param size_t po2 - power of 2 that used in rice code
* @param int delta - integer value to encode
* @param bit_stream* bs - pointer to bit_stream where to write a result of encoding 
*/

void  encode_rice(size_t po2, int delta, bit_stream* bs);

/**
* read_loss
* @breif This function reads from file loss (BUFFER_LEN amount of loss) and also scale
* @param FILE* file - file to read loss
* @param float* loss - pointer to an array where to write loss
* @param float* scale - pointer to an float variable where to save scale
*/

void read_loss(FILE* file, float* loss, float* scale);

/**
* bit_stream_set
* @breif Sets a value of bit (1 or 0) into bit_stream structure
* @param bit_stream* bs - pointer to a bit_stream to set a bit
* @param int set - value to set
*/

void  bit_stream_set(bit_stream* bs, int set);

/**
* bit_stream_init
* @breif Inits bit_stream structure
* @param bit_stream* bs - pointer to an bit_stream structure to init
*/

void bit_stream_init(bit_stream* bs);
