#include <cmath>
#include <stdbool.h>

void encode_rice(int k, int number,int* p,int* q, bool* sign);
void decode_rice(int k, int* number, int p, int q, bool* sign);

void encode_rice(int k, int number,int* p,int* q, bool* sign)
{
    if (number < 0)
        *sign = 1;
    else
        *sign = 0; 

    *p = abs(number) / k;
    *q = abs(number) - k*(*p);  
}

void decode_rice(int k, int* number, int p, int q, bool* sign)
{
    *number = p*k + q;
    if (sign)
        *number = *number*(-1);
}