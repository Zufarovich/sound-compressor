#include <stdio.h>
#include <stdlib.h>

#define BUTCH_CONSTANT 1024


typedef struct _bin_arr{
    char* bf;
    int position;
    int size;
} bin_arr;


void init_bin_arr(bin_arr* ba){
    ba->bf = (char*)malloc(BUTCH_CONSTANT);
    ba->position = 0;
    ba->size = BUTCH_CONSTANT;
}


void set_bin_arr(bin_arr* ba, int not_zero){
    int byte = ba->position / 8;
    int bit = ba->position % 8;

    if(ba->position == ba->size)
        ba->bf = (char*)realloc(ba->bf, ba->size *= 2);

    ba->bf[byte] = not_zero ? ba->bf[byte] | (1<<(7-bit)) : ba->bf[byte] & ~(1<<(7-bit));
    ba->position++;
}


void fill_bin_arr(int rem_len, int po2, int delta, bin_arr* ba){
    if (!delta) set_bin_arr(ba, 0);
    else{
        set_bin_arr(ba, 1);
        set_bin_arr(ba, delta<0);
        delta = abs(delta);
        int k = delta;
        while(k >= po2){
            set_bin_arr(ba, 1);
            k-=po2;
        }
        set_bin_arr(ba, 0); 
        int shift = 0;
        while(rem_len){
            rem_len--;
            set_bin_arr(ba, delta & 1<<rem_len);
        }          
    }
}


void BEwriter(FILE* file, double m, int po2, int* arr){
    int num = 1;
    int rem_len = 1;
    while((num*=2)<po2) rem_len++;

    bin_arr ba;
    init_bin_arr(&ba);

    char* num_to_bin = (char*)&m;
    for(int i = 0; i<sizeof(double); i++) 
        fputc(num_to_bin[i], file);
    num_to_bin = (char*)&po2;
    for(int i = 0; i<sizeof(int); i++) 
        fputc(num_to_bin[i], file);

    for(int i = 0; i < BUTCH_CONSTANT; i++)
        fill_bin_arr(rem_len, po2, arr[i], &ba);

    int limit = ba.position;
    for(int i = 0; i*8 < num; i++)
        fputc(ba.bf[i], file);
    
    free(ba.bf);
}