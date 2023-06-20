/**
*   encode_rice 
*   This function accepts int k - power of 2 to use it in encoding, int number - number to encode.
*   Also, it accepts pointer to int p and q (2 parameters after encoding).
*   Bool sign indicates, whether the number is negative.(True if it's negative, false otherwise).
*   @param int k - power of 2 to use it in encoding
*   @param int number - number to encode
*   @param int* p - integer part of division in rice encoding
*   @param int* q - remainder of the division in rice encoding
*   @param bool* sign - true if number is negative, false otherwise
*/
void encode_rice(int k, int number,int* p,int* q, bool* sign);

/**
*   decode_rice 
*   This function accepts int k - power of 2 to use it in decoding, int* number - result of decoding.
*   Also, it accepts int p and q (2 parameters to use in rice decoding).
*   Bool sign indicates, whether the number is negative.(True if it's negative, false otherwise).
*   @param int k - power of 2 to use it in decoding
*   @param int number - result of decoding
*   @param int p - integer part of division in rice encoding
*   @param int q - remainder of the division in rice encoding
*   @param bool* sign - true if number is negative, false otherwise
*/
void decode_rice(int k, int* number, int p, int q, bool* sign);