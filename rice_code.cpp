#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "rice_code.h"

#define check_end if (!(position % 8)) readcount = fread(&buffer, sizeof(char), 1, file)
#define bit_readen (buffer & (1 << (position % 8)))

void bit_stream_init(bit_stream* bs) {
    bs->buf = (char*)malloc(1024); 
    memset(bs->buf, '\0', 1024);
    bs->cap = 1024;
    bs->bit = 0;
}

void bit_stream_set(bit_stream* bs, int set) {
    size_t byte = bs->bit / 8;
    size_t bit = bs->bit  % 8;
    if ( byte >= bs->cap )
    {  
        bs->buf = (char*)realloc(bs->buf, bs->cap *= 2);
        memset(bs->buf + bs->cap/2, '\0', bs->cap/2);
    }
    bs->buf[byte] = set ? bs->buf[byte] | (1<<bit) : bs->buf[byte] & ~(1<<bit);
    bs->bit++;
}

void encode_rice(size_t po2, int delta, bit_stream* bs) {
    bit_stream_set(bs, delta > 0);
    int k = abs(delta);
    while ( k > 0 && k >= po2) {
        bit_stream_set(bs, 1);
        k -= po2;
    }
    bit_stream_set(bs, 0);
    int b = 0;
    while (po2 - 1) {
        bit_stream_set(bs, abs(delta) & (1<<b));
        po2 /= 2;
        b++;        
    }
}

void read_loss(FILE* file, float* loss, float* scale)
{
    int     count, position, readcount;
    char    buffer = 0;
    short po2;

    po2 = count = position = 0;

    fread(&po2, sizeof(short)  , 1, file);
    fread(scale, sizeof(float), 1, file);

    while (count < BUFFER_LEN && readcount)
    {   
        check_end;

        int above_zero =  bit_readen ?  1 : -1;
        
        position++;
        check_end;

        int amount_of_one = 0;

        while (bit_readen && readcount)
        {
            amount_of_one++;
            position++;

            check_end;
        }

        int current_po2 = 1;
        short save_po2 = po2;
        int modulo = 0;

        while (save_po2 - 1 && readcount)
        {
            position++;
            check_end;

            if (bit_readen)
                modulo += current_po2;

            current_po2 *= 2;

            save_po2 /= 2;
        }

        loss[count] = (po2*amount_of_one + modulo)*above_zero;

        count++;
        position++;
    }
}

void print_loss(FILE* file, short po2, bit_stream* bs, float scale)
{   
    fwrite(&po2, sizeof(short), 1, file);
    fwrite(&scale, sizeof(float), 1, file);

    size_t byte_to_read = bs->bit % 8 ? bs->bit / 8 + 1 : bs->bit / 8;

    for (int i = 0; i < byte_to_read; i++)
        fputc(bs->buf[i], file);
}
