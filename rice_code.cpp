#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#define check_end if (!(position % 8)) fread(&buffer, sizeof(char), 1, file)
#define bit_readen (buffer & (1 << (position % 8)))
#define BUFFER_LEN 1024

typedef struct _bit_stream {
    char*  buf;
    size_t len;
    size_t cap;
    size_t bit;
} bit_stream;

void bit_stream_init(bit_stream* bs) {
    bs->buf = (char*)malloc(1024);
    memset(bs->buf, '\0', 1024);
    bs->len = 0;
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
    if ( bs->bit > bs->len) bs->len = bs->bit;
    bs->bit++;
}

void encode_rice(size_t po2, int delta, bit_stream* bs) {
    if ( delta == 0) {
        bit_stream_set(bs, 0);
    } else {
        bit_stream_set(bs, 1);
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
}

void read_loss(FILE* file, float* scale, float* loss)
{
    float m = 0;
    int   po2, count, position;
    char  buffer = 0;

    po2 = count = position = 0;

    fread(&m  , sizeof(float), 1, file);
    fread(&po2, sizeof(int)  , 1, file);
    printf("%d\n", po2);

    while (count < BUFFER_LEN)
    {   
        check_end;

        if (!bit_readen)
        {
            loss[count] = 0;
            position++;
            count++;
        }
        else
        {
            position++;
            check_end;

            int above_zero =  bit_readen ?  1 : -1;
            
            position++;
            check_end;

            int amount_of_one = 0;

            while (bit_readen)
            {
                amount_of_one++;
                position++;

                check_end;
            }

            int current_po2 = 1;
            int save_po2 = po2;
            int modulo = 0;

            while (save_po2 - 1)
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
}

void print_loss(FILE* file, float scale, int po2, bit_stream* bs)
{   
    fwrite(&scale, sizeof(float), 1, file);
    fwrite(&po2, sizeof(int), 1, file);

    size_t byte_to_read = bs->len % 8 ? bs->len / 8 + 1 : bs->len / 8;

    for (int i = 0; i < byte_to_read; i++)
        fputc(bs->buf[i], file);
}
