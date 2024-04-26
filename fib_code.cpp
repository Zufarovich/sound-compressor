#include <iostream>
#include "rice_code.h"
#include <stdio.h>
#include <cmath>

FILE* log_file = fopen("log.txt", "w");

const int num_fib[] = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377};

#define check_end if (!(position % 8)) readcount = fread(&buffer, sizeof(char), 1, file)
#define bit_readen (buffer & (1 << (position % 8)))

int pow_int(int basis, int power)
{
    int result = 1;

    for(int i = 0; i < power; i++)
        result *= basis;

    return result;
}

bit_stream* num_to_fib(bit_stream* bs, int num)
{
    bool already_set = false;

    if(num >= 0)
    {
        bit_stream_set(bs, 1);
        //fprintf(log_file, "1");
    }
    else
    {
        bit_stream_set(bs, 0);
        num *= -1;
        //fprintf(log_file, "0");
    }

    for(int i = int(sizeof(num_fib)/sizeof(int)) - 1; i >= 0 || num > 0; i--)
    {
        //printf("num:%d\tnum_fib:%d\ti:%d\n", num, num_fib[i], i);
        if(already_set && num < num_fib[i])
        {
            bit_stream_set(bs, 0);
            //fprintf(log_file, "0");
        }
        else if(num >= num_fib[i])
        {
            bit_stream_set(bs, 1);
            //fprintf(log_file, "1");
            num -= num_fib[i];
            already_set = true;
        }
    }

    bit_stream_set(bs, 1);
    bit_stream_set(bs, 1);
    //fprintf(log_file, "11");

    return bs;
}

void read_loss_fib(FILE* file, short* loss)
{
    int     count, position, readcount;
    char    buffer = 0;
    int     len    = 0;
    int     previous;
    int     save[sizeof(num_fib)/sizeof(int)];

    count = position = 0;

    //write reading scale
    readcount = 1;

    while (count < 2 && readcount)
    {   
        check_end;

        int above_zero =  bit_readen ? 1 : -1;

        fprintf(log_file, "%d ", above_zero);
        
        position++;
        check_end;

        bool not_end_of_num = true;
        previous = 0;
        len = 0;

        while(not_end_of_num)
        {
            if(previous && bit_readen)
            {
                not_end_of_num = false;
                len--;
            }
            else if(bit_readen && !previous)
            {
                save[len] = 1;
                previous = 1;
                fprintf(log_file, "%d", save[len]);
                len++;
            }
            else 
            {
                save[len] = 0;
                previous = 0;
                fprintf(log_file, "%d", save[len]);
                len++;
            }
            
            position++;
            check_end;
        }

        count++;
    }
}

void num_to_gamma_code(bit_stream* bs, int num)
{
    
    if(num >= 0)
        bit_stream_set(bs, 1);
    else
    {
        bit_stream_set(bs, 0);
        num *= -1;
    }

    int max_pow = 0;

    while(num / pow_int(2, max_pow))
        max_pow++;

    if(!max_pow)
        bit_stream_set(bs, 1);
    
    for(int i = 0; i < max_pow; i++)
        bit_stream_set(bs, 0);

    for(int i = max_pow - 1; i >= 0; i--)
    {
        if(num < pow_int(2, i))
            bit_stream_set(bs, 0);
        else
        {
            bit_stream_set(bs, 1);
            num -= pow_int(2, i);
        }
    }
}

void read_loss_gamma(FILE* file, short* loss)
{
    int     count, position, readcount;
    char    buffer = 0;
    int     len    = 0;
    int     zeros  = 0;

    count = position = 0;

    //write reading scale
    readcount = 1;

    while (count < 4 && readcount)
    {   
        check_end;

        int above_zero =  bit_readen ? 1 : -1;

        position++;
        check_end;

        while(!bit_readen)
        {
            position++;
            check_end;
            zeros++;
        }

        printf("%d\n", zeros);

        if(zeros)
        {
            loss[count] = pow_int(2, zeros - 1);
            position++;
            check_end;
        }
        else
        {
            position++;
            check_end;
            loss[count] = 0;
            count++;
            continue;
        }
        
        for(int i = zeros - 2; i >= 0; i--)
        {
            if(bit_readen)
                loss[count] += pow_int(2, i);

            position++;
            check_end;
            printf("%d %d\n", i, zeros);
        }

        loss[count] *= above_zero;

        count++;
        zeros = 0;
    }
}

int main()
{
    bit_stream bs;
    bit_stream_init(&bs);

   // num_to_fib(&bs, 7);
    //printf("\n");
    //num_to_fib(&bs, -9);
    
    num_to_gamma_code(&bs, -21);
    num_to_gamma_code(&bs, 0);
    num_to_gamma_code(&bs, 17);
    num_to_gamma_code(&bs, 10);

    FILE* file = fopen("test.nlac", "w");

    size_t byte_to_read = bs.bit % 8 ? bs.bit / 8 + 1 : bs.bit / 8;

    for (int i = 0; i < byte_to_read; i++)
        fputc(bs.buf[i], file);

    fclose(file);

    free(bs.buf);

    FILE* reading = fopen("test.nlac", "r");

    short test[4] = {0};

    read_loss_gamma(reading, test);
    printf("%d %d %d %d\n", test[0], test[1], test[2], test[3]);

    fclose(reading);
    fclose(log_file);
}