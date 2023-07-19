#pragma GCC optimize("O3")
#pragma GCC target("avx2")

#include <stdio.h>
#include <sndfile.h>
#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits.h>
#include <stdbool.h>
#include <torch/script.h>
#include <vector>
#include "rice_code.h"

#define DISCRETE_FACTOR_16 SHRT_MAX
#define BUFFER_LEN 2048
#define CROSS_FADE 204
#define COMPRESSED_PARAMETERS 64
#define MAX_CHANNELS 6
#define ERROR_OPEN_INPUT -1
#define ERROR_OPEN_OUTPUT -2
#define EXTRA_CHANNELS -3
#define LACK_OF_FILES -4
#define _USE_MATH_DEFINES

SNDFILE* SIGNAL;
SNDFILE* WRITE;
SF_INFO  SFINFO_WRITE;
SF_INFO  SFINFO_SIGNAL;

void  process_channel(float* data, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder, FILE* write);
void  process_data_encode(char* input, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder);
void  decode_sample(float* result, float* input, float max_ampl, torch::jit::script::Module* decoder);
void  process_data_decode(SNDFILE* output, FILE* file_to_decode, torch::jit::script::Module* module);
void  save_loss(int k, float* buffer, float* data, float* data1, bit_stream* bs, FILE* to_write);
void  window_hann(float* data, float* transformed_data, int window_size);
int   open_sf_write(char* file, SNDFILE** sf_file, SF_INFO* sf_file_info);
int   open_sf_read(char* file, SNDFILE** sf_file, SF_INFO* sf_file_info);
void  move_second_part(float* data, size_t len);
void  move_third_part(float* data, size_t len);
float find_max(float* loss, size_t len);
int   find_mean(int* data, size_t len);
void  bit_stream_init(bit_stream* bs);
void  unzip(float* data, size_t len);
void  zip(float* data, size_t len);

void process_data_encode_new(char* input, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder);
void save_loss_new(float* data, float* data1, bit_stream* bs, FILE* to_write);
void process_data_decode_new(SNDFILE* output, FILE* file_to_decode, torch::jit::script::Module* decoder);
void decode_sample_new(float* result, float* input, float max_ampl, torch::jit::script::Module* decoder);

void process_data_encode_cross_fade(char* input, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder);
void process_data_decode_cross_fade(SNDFILE* output, FILE* file_to_decode, torch::jit::script::Module* decoder);
void save_loss_cross_fade(int k, float* buffer, float* data, float* data1, bit_stream* bs, FILE* to_write);

int main(int argc, char* argv[])
{
    SIGNAL = NULL;
    memset(&SFINFO_SIGNAL, 0, sizeof(SFINFO_SIGNAL));

    if(argc > 1)
    {
        torch::NoGradGuard no_grad;
        torch::jit::script::Module decoder = torch::jit::load("decoder1.pth");

        if (!strcmp(argv[1], "-e")){
            torch::jit::script::Module encoder = torch::jit::load("encoder1.pth");

            encoder.eval();
            decoder.eval();

            if (argc > 2){
                process_data_encode_cross_fade(argv[2], &encoder, &decoder);
            }
            else{
                printf("You have to enter also file_name to encode!\n");
                return LACK_OF_FILES;
            }
        }
        else if (!strcmp(argv[1], "-d")){
            decoder.eval();

            if (argc > 3){
                FILE* decode = fopen(argv[2], "r");
                open_sf_write(argv[3], &WRITE, &SFINFO_WRITE);
                process_data_decode_cross_fade(WRITE, decode, &decoder);
                fclose(decode);
            }
            else{
                printf("You have to enter also file_name to decode and file_name to write decoded file!\n");
                return LACK_OF_FILES;
            }
        }
        else if (!strcmp(argv[1], "-h")){
            printf("Enter -e and signal file_name to encode it.\n");
            printf("Enter -d, file_name to decode and file_name to write decoded file.\n");
        }
        else{
            printf("Enter -h to get help.\n");
        }

        return 0;
    }
    else
    {
        printf("Enter -h to get help.\n");
        return LACK_OF_FILES;
    }
}

int open_sf_read(char* file, SNDFILE** sf_file, SF_INFO* sf_file_info)
{
    if (!(*sf_file = sf_open(file, SFM_READ, sf_file_info)))
    {
        printf("%s\n", sf_strerror(*sf_file));
        printf("Unable to open the input file\n");
        return ERROR_OPEN_INPUT;
    }

    if (sf_file_info->channels > MAX_CHANNELS)
    {
        printf("Not able to process more than %d channels\n", MAX_CHANNELS);
        return EXTRA_CHANNELS;
    }
    
    return 0;
}

int open_sf_write(char* file, SNDFILE** sf_file, SF_INFO* sf_file_info)
{
    memset(sf_file_info, 0, sizeof(*sf_file_info));

    sf_file_info->format = SF_FORMAT_WAV | SF_FORMAT_PCM_32;
    sf_file_info->channels = 2;
    sf_file_info->samplerate = 44100;

    if (!(*sf_file = sf_open(file, SFM_WRITE, sf_file_info)))
    {
        printf("%s\n", sf_strerror(*sf_file));
        printf("Unable to open the input file\n");
        return ERROR_OPEN_INPUT;
    }

    return 0;
}

int find_mean(int* data, size_t len)
{
    long long sum = 0;

    for (int i = 0; i < len; i++)
        sum += abs(data[i]);

    return sum/len;
}

float find_max(float* loss, size_t len)
{
    float maximum = 0;

    for (int i = 0; i < len; i++)
    {
        if(fabs(loss[i]) > maximum)
            maximum = fabs(loss[i]);
    }

    return maximum;
}

void window_hann(float* data, float* transformed_data, int window_size)
{
    for (int i = 0; i < window_size; i++)
        transformed_data[i] = 0.5 * (1 - cos((2 * M_PI * i)/(window_size - 1))) * data[i];
}

void zip(float* data, size_t len)
{
    float buf1[len/2];
    float buf2[len/2];

    for (int i = 0; i < len/2; i++)
    {
        buf1[i] = data[2*i];
        buf2[i] = data[2*i + 1];
    }

    for (int i = 0; i < len/2; i++)
    {
        data[i] = buf1[i];
        data[len/2 + i] = buf2[i];
    }
}

void unzip(float* data, size_t len)
{
    float buf1[len/2];
    float buf2[len/2];

    for (int i = 0; i < len/2; i++)
    {
        buf1[i] = data[i];
        buf2[i] = data[len/2 + i];
    }

    for (int i = 0; i < len/2; i++)
    {
        data[2*i] = buf1[i];
        data[2*i + 1] = buf2[i];
    }
}

void move_second_part(float* data, size_t len)
{
    memcpy(data, data + len/2, sizeof(float)*len/2);
}

void move_third_part(float* data, size_t len)
{
    memcpy(data, data + 2*len/3, sizeof(float)*len/3);

    memset(data + len/3, 0, 2*len/3*sizeof(float));
}

void process_channel(float* data, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder, FILE* write)
{
    torch::Tensor inputs = torch::empty({1, BUFFER_LEN});
    torch::Tensor encoded = torch::empty({1, COMPRESSED_PARAMETERS});

    float max_ampl = find_max(data, BUFFER_LEN);

    if (max_ampl != 0)
    {
        for (int i = 0; i < BUFFER_LEN; i++)
            data[i] /= max_ampl;

        window_hann(data, data, BUFFER_LEN);

        for (int i = 0; i < BUFFER_LEN; i++)
            inputs[0][i] = data[i];

        auto params = encoder->forward({inputs}).toTensor();

        for (int i = 0; i < COMPRESSED_PARAMETERS; i++)
            data[i] = params[0][i].item<float>();

        fwrite(data     , sizeof(float), COMPRESSED_PARAMETERS, write); 
        fwrite(&max_ampl, sizeof(float), 1                    , write); 

        for (int i = 0; i < COMPRESSED_PARAMETERS; i++)
            encoded[0][i] = data[i];

        auto output = decoder->forward({encoded}).toTensor();

        for (int i = 0; i < BUFFER_LEN; i++)
            data[i] = output[0][i].item<float>() * max_ampl;
    }
    else
    {
        for (int i = 0; i < BUFFER_LEN; i++)
            data[i] = 0;

        fwrite(data     , sizeof(float), COMPRESSED_PARAMETERS, write);
        fwrite(&max_ampl, sizeof(float), 1                    , write);
    }
}

void write_loss(float* data, bit_stream* bs, FILE* output)
{
    int pow_of_2;
    int loss [BUFFER_LEN];

    float max_loss = find_max(data, BUFFER_LEN);

    if (!max_loss)
    {
        for (int i = 0; i < BUFFER_LEN; i++)
            loss[i] = 0;
    }
    else
    {
        for(int i = 0; i < BUFFER_LEN; i++) 
            loss[i] = data[i] * DISCRETE_FACTOR_16 ;
    } 

    /*int mean = find_mean(loss, BUFFER_LEN);
    pow_of_2 = 2;
    int delta = 0;

    while (delta >= 0 && mean != 0)
    {
        delta = 0;

        for (int i = 0; i < BUFFER_LEN; i++)
            encode_rice(pow_of_2, loss[i], bs);
        delta += bs-> bit;
        bs->bit = 0;
        bs->len = 0;

        pow_of_2 *= 2;
        for (int i = 0; i < BUFFER_LEN; i++)
            encode_rice(pow_of_2, loss[i], bs);
        delta -= bs->bit;

        bs->bit = 0;
        bs->len = 0;
    }*/
    
    pow_of_2 = 1 << (sizeof(int)*8 - __builtin_clz(find_mean(loss, BUFFER_LEN)));
    //pow_of_2 /= 2;

    for (int i = 0; i < BUFFER_LEN; i++)
        encode_rice(pow_of_2, loss[i], bs);

    print_loss(output, pow_of_2, bs);
}

void decode_sample(float* result, float* input, float max_ampl, torch::jit::script::Module* decoder)
{
    torch::Tensor inputs = torch::empty({1, COMPRESSED_PARAMETERS});

    for (int i = 0; i < COMPRESSED_PARAMETERS; i++)
        inputs[0][i] = input[i];

    auto output = decoder->forward({inputs}).toTensor();

    for (int i = 0; i < BUFFER_LEN; i++)
        result[i] += output[0][i].item<float>() * max_ampl;
}

void save_loss(int k, float* buffer, float* data, float* data1, bit_stream* bs, FILE* to_write)
{
    if ((k % 2))
    {
        for (int i = 0; i < BUFFER_LEN; i++)
            buffer[i] += data[i];
    }
    else
    {
        for (int i = 0; i < BUFFER_LEN; i++)
            buffer[(BUFFER_LEN/2) + i] += data[i];

        float loss[BUFFER_LEN];

        for (int i = 0; i < BUFFER_LEN; i++)
            loss[i] = buffer[i] - data1[i];

        write_loss(loss, bs, to_write);
        move_third_part(buffer, 3*BUFFER_LEN/2);

        bs->bit = 0;
        bs->len = 0;
    }       
}

void process_data_encode(char* input, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder)
{
    int    k = 0;
    int    size_to_read = BUFFER_LEN;
    float  data1   [3*BUFFER_LEN]   = {0};
    float  data2   [BUFFER_LEN];
    float  buffer1 [3*BUFFER_LEN/2] = {0};
    float  buffer2 [3*BUFFER_LEN/2] = {0};
    float* place_to_read = data1;

    open_sf_read (input, &SIGNAL, &SFINFO_SIGNAL);
    char output_file[strchr(input, '.') - input + 5];
    strcpy(output_file                 , input );
    strcpy(strchr(output_file, '.') + 1, "nlac");

    FILE* music = fopen(output_file, "wb");

    while (sf_readf_float(SIGNAL, place_to_read, size_to_read))
    {   
        bit_stream bs;
        bit_stream_init(&bs);
        k++;

        zip(data1, sizeof(data1)/sizeof(float));

        memcpy(data2, data1 + ((k + 1) % 2)*BUFFER_LEN/2                 , BUFFER_LEN*sizeof(float));
        process_channel(data2, encoder, decoder, music);
        save_loss(k, buffer1, data2, data1                 , &bs, music);     

        memcpy(data2, data1 + 3*BUFFER_LEN/2 + ((k + 1) % 2)*BUFFER_LEN/2, BUFFER_LEN*sizeof(float));
        process_channel(data2, encoder, decoder, music); 
        save_loss(k, buffer2, data2, data1 + 3*BUFFER_LEN/2, &bs, music);     

        unzip(data1, sizeof(data1)/sizeof(float));

        if (k % 2)
        {
            place_to_read = data1 + 2*BUFFER_LEN;
        }
        else
        {
            move_third_part(data1, 3*BUFFER_LEN);
            place_to_read = data1 + BUFFER_LEN;
        }

        if(k == 1)
            size_to_read /= 2;

        free(bs.buf);
    }

    sf_close(SIGNAL);
}

void process_data_decode(SNDFILE* output, FILE* file_to_decode, torch::jit::script::Module* decoder)
{
    int readcount = 1;

    float buf_channel1  [2*COMPRESSED_PARAMETERS];
    float buf_channel2  [2*COMPRESSED_PARAMETERS];
    float saved_channel1[3*BUFFER_LEN/2] = {0};
    float saved_channel2[3*BUFFER_LEN/2] = {0};
    float save          [2*BUFFER_LEN  ] = {0};

    while(readcount)
    {
        float max_loss_ch1_1, max_loss_ch1_2, max_loss_ch2_1, max_loss_ch2_2;
        float loss1[BUFFER_LEN];
        float loss2[BUFFER_LEN];

        readcount = fread(buf_channel1,                         sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch1_1,                      sizeof(float), 1                    , file_to_decode);
        readcount = fread(buf_channel2,                         sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch2_1,                      sizeof(float), 1                    , file_to_decode);
        readcount = fread(buf_channel1 + COMPRESSED_PARAMETERS, sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch1_2,                      sizeof(float), 1                    , file_to_decode);

        read_loss(file_to_decode, loss1);

        readcount = fread(buf_channel2 + COMPRESSED_PARAMETERS, sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch2_2,                      sizeof(float), 1                    , file_to_decode);

        read_loss(file_to_decode, loss2);
        
        decode_sample(saved_channel1               , buf_channel1                        , max_loss_ch1_1, decoder);
        decode_sample(saved_channel2               , buf_channel2                        , max_loss_ch2_1, decoder);
        decode_sample(saved_channel1 + BUFFER_LEN/2, buf_channel1 + COMPRESSED_PARAMETERS, max_loss_ch1_2, decoder);
        decode_sample(saved_channel2 + BUFFER_LEN/2, buf_channel2 + COMPRESSED_PARAMETERS, max_loss_ch2_2, decoder);

        for (int i = 0; i < BUFFER_LEN; i++)
        {
            loss1[i] = loss1[i] / DISCRETE_FACTOR_16;
            loss2[i] = loss2[i] / DISCRETE_FACTOR_16;
        }

        for (int i = 0; i < BUFFER_LEN; i++)
        {
            saved_channel1[i] -= loss1[i];
            saved_channel2[i] -= loss2[i];
            save[2*i]          = saved_channel1[i];
            save[2*i + 1]      = saved_channel2[i];
        }

        move_third_part(saved_channel1, 3*BUFFER_LEN/2);
        move_third_part(saved_channel2, 3*BUFFER_LEN/2);

        sf_writef_float(output, save, BUFFER_LEN);
    }

    sf_close(output);
}

void process_data_encode_new(char* input, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder)
{
    float  data1   [2*BUFFER_LEN] = {0};
    float  data2   [BUFFER_LEN];
    float  buffer1 [BUFFER_LEN] = {0};
    float  buffer2 [BUFFER_LEN] = {0};

    open_sf_read (input, &SIGNAL, &SFINFO_SIGNAL);
    char output_file[strchr(input, '.') - input + 5];
    strcpy(output_file                 , input );
    strcpy(strchr(output_file, '.') + 1, "nlac");

    FILE* music = fopen(output_file, "wb");

    while (sf_readf_float(SIGNAL, data1, BUFFER_LEN))
    {   
        bit_stream bs;
        bit_stream_init(&bs);

        zip(data1, sizeof(data1)/sizeof(float));

        memcpy(data2, data1 , BUFFER_LEN*sizeof(float));
        process_channel(data2, encoder, decoder, music);
        save_loss_new(data2, data1                 , &bs, music);     

        memcpy(data2, data1 + BUFFER_LEN , BUFFER_LEN*sizeof(float));
        process_channel(data2, encoder, decoder, music); 
        save_loss_new(data2, data1 + BUFFER_LEN, &bs, music);     

        free(bs.buf);
    }

    sf_close(SIGNAL);
}

void save_loss_new(float* data, float* data1, bit_stream* bs, FILE* to_write)
{
    float loss[BUFFER_LEN];

    for (int i = 0; i < BUFFER_LEN; i++)
        loss[i] = data[i] - data1[i];

    write_loss(loss, bs, to_write);

    bs->bit = 0;
    bs->len = 0;       
}

void process_data_decode_new(SNDFILE* output, FILE* file_to_decode, torch::jit::script::Module* decoder)
{
    int readcount = 1;

    float buf_channel1  [2*COMPRESSED_PARAMETERS];
    float buf_channel2  [2*COMPRESSED_PARAMETERS];
    float saved_channel1[3*BUFFER_LEN/2] = {0};
    float saved_channel2[3*BUFFER_LEN/2] = {0};
    float save          [2*BUFFER_LEN  ] = {0};

    while(readcount)
    {
        float max_loss_ch1, max_loss_ch2;
        float loss1[BUFFER_LEN];
        float loss2[BUFFER_LEN];

        readcount = fread(buf_channel1,                         sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch1,                        sizeof(float), 1                    , file_to_decode);

        read_loss(file_to_decode, loss1);

        readcount = fread(buf_channel2,                         sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch2,                        sizeof(float), 1                    , file_to_decode);

        read_loss(file_to_decode, loss2);
        
        decode_sample_new(saved_channel1               , buf_channel1                        , max_loss_ch1, decoder);
        decode_sample_new(saved_channel2               , buf_channel2                        , max_loss_ch2, decoder);

        for (int i = 0; i < BUFFER_LEN; i++)
        {
            loss1[i] = loss1[i] / DISCRETE_FACTOR_16;
            loss2[i] = loss2[i] / DISCRETE_FACTOR_16;
        }

        for (int i = 0; i < BUFFER_LEN; i++)
        {
            saved_channel1[i] -= loss1[i];
            saved_channel2[i] -= loss2[i];
            save[2*i]          = saved_channel1[i];
            save[2*i + 1]      = saved_channel2[i];
        }

        sf_writef_float(output, save, BUFFER_LEN);
    }

    sf_close(output);
}

void decode_sample_new(float* result, float* input, float max_ampl, torch::jit::script::Module* decoder)
{
    torch::Tensor inputs = torch::empty({1, COMPRESSED_PARAMETERS});

    for (int i = 0; i < COMPRESSED_PARAMETERS; i++)
        inputs[0][i] = input[i];

    auto output = decoder->forward({inputs}).toTensor();

    for (int i = 0; i < BUFFER_LEN; i++)
        result[i] = output[0][i].item<float>() * max_ampl;
}

void process_data_encode_cross_fade(char* input, torch::jit::script::Module* encoder, torch::jit::script::Module* decoder)
{
    int    k = 0;
    float  data1   [4*BUFFER_LEN]   = {0};
    float  data2   [BUFFER_LEN];
    float  buffer1 [2*BUFFER_LEN - CROSS_FADE] = {0};
    float  buffer2 [2*BUFFER_LEN - CROSS_FADE] = {0};
    float* place_to_read = data1;

    open_sf_read (input, &SIGNAL, &SFINFO_SIGNAL);
    char output_file[strchr(input, '.') - input + 5];
    strcpy(output_file                 , input );
    strcpy(strchr(output_file, '.') + 1, "nlac");

    FILE* music = fopen(output_file, "wb");

    while (sf_readf_float(SIGNAL, place_to_read, BUFFER_LEN))
    {   
        bit_stream bs;
        bit_stream_init(&bs);
        k++;

        zip(data1, sizeof(data1)/sizeof(float));

        memcpy(data2, data1 + ((k + 1) % 2)*BUFFER_LEN               , BUFFER_LEN*sizeof(float));
        process_channel(data2, encoder, decoder, music);
        save_loss_cross_fade(k, buffer1, data2, data1                              , &bs, music);     

        memcpy(data2, data1 + 2*BUFFER_LEN + ((k + 1) % 2)*BUFFER_LEN, BUFFER_LEN*sizeof(float));
        process_channel(data2, encoder, decoder, music); 
        save_loss_cross_fade(k, buffer2, data2, data1 + 2*BUFFER_LEN               , &bs, music);     

        unzip(data1, sizeof(data1)/sizeof(float));

        if (k != 1)
        {
            memcpy(data1, data1 + 2*BUFFER_LEN, 2*BUFFER_LEN*sizeof(float));
            place_to_read = data1 + 2*BUFFER_LEN;
        }

        free(bs.buf);
    }

    sf_close(SIGNAL);
}

void save_loss_cross_fade(int k, float* buffer, float* data, float* data1, bit_stream* bs, FILE* to_write)
{
    if (k == 1)
    {
        for (int i = 0; i < BUFFER_LEN; i++)
            buffer[i] += data[i];
    }
    else
    {
        for (int i = 0; i < BUFFER_LEN; i++)
            buffer[BUFFER_LEN - CROSS_FADE + i] += data[i];

        float loss[BUFFER_LEN];

        for (int i = 0; i < BUFFER_LEN; i++)
            loss[i] = buffer[i] - data1[i];

        write_loss(loss, bs, to_write);
        memcpy(buffer, buffer + BUFFER_LEN - CROSS_FADE, sizeof(float)*BUFFER_LEN);
        memset(buffer + BUFFER_LEN, 0, sizeof(float)*(BUFFER_LEN - CROSS_FADE));

        bs->bit = 0;
        bs->len = 0;
    }
}

void process_data_decode_cross_fade(SNDFILE* output, FILE* file_to_decode, torch::jit::script::Module* decoder)
{
    int readcount = 1;

    float buf_channel1  [2*COMPRESSED_PARAMETERS];
    float buf_channel2  [2*COMPRESSED_PARAMETERS];
    float saved_channel1[2*BUFFER_LEN - CROSS_FADE] = {0};
    float saved_channel2[2*BUFFER_LEN - CROSS_FADE] = {0};
    float save          [2*BUFFER_LEN] = {0};
    float loss1         [BUFFER_LEN];
    float loss2         [BUFFER_LEN];
    float max_loss_ch1_1, max_loss_ch1_2, max_loss_ch2_1, max_loss_ch2_2;

    readcount = fread(buf_channel1,                         sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
    readcount = fread(&max_loss_ch1_1,                      sizeof(float), 1                    , file_to_decode);
    readcount = fread(buf_channel2,                         sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
    readcount = fread(&max_loss_ch2_1,                      sizeof(float), 1                    , file_to_decode);
    readcount = fread(buf_channel1 + COMPRESSED_PARAMETERS, sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
    readcount = fread(&max_loss_ch1_2,                      sizeof(float), 1                    , file_to_decode);

    read_loss(file_to_decode, loss1);

    readcount = fread(buf_channel2 + COMPRESSED_PARAMETERS, sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
    readcount = fread(&max_loss_ch2_2,                      sizeof(float), 1                    , file_to_decode);

    read_loss(file_to_decode, loss2);
    
    decode_sample(saved_channel1                          , buf_channel1                        , max_loss_ch1_1, decoder);
    decode_sample(saved_channel2                          , buf_channel2                        , max_loss_ch2_1, decoder);
    decode_sample(saved_channel1 + BUFFER_LEN - CROSS_FADE, buf_channel1 + COMPRESSED_PARAMETERS, max_loss_ch1_2, decoder);
    decode_sample(saved_channel2 + BUFFER_LEN - CROSS_FADE, buf_channel2 + COMPRESSED_PARAMETERS, max_loss_ch2_2, decoder);

    for (int i = 0; i < BUFFER_LEN; i++)
    {
        loss1[i] = loss1[i] / DISCRETE_FACTOR_16;
        loss2[i] = loss2[i] / DISCRETE_FACTOR_16;
    }

    for (int i = 0; i < BUFFER_LEN; i++)
    {
        save[2*i]          = saved_channel1[i] - loss1[i];
        save[2*i + 1]      = saved_channel2[i] - loss2[i];
    }

    memcpy(saved_channel1, saved_channel1 + BUFFER_LEN - CROSS_FADE, sizeof(float)*BUFFER_LEN);
    memset(saved_channel1 + BUFFER_LEN, 0, sizeof(float)*(BUFFER_LEN - CROSS_FADE));
    memcpy(saved_channel2, saved_channel2 + BUFFER_LEN - CROSS_FADE, sizeof(float)*BUFFER_LEN);
    memset(saved_channel2 + BUFFER_LEN, 0, sizeof(float)*(BUFFER_LEN - CROSS_FADE));

    sf_writef_float(output, save, BUFFER_LEN);

    while(readcount)
    {
        readcount = fread(buf_channel1 + COMPRESSED_PARAMETERS, sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch1_2,                      sizeof(float), 1                    , file_to_decode);

        read_loss(file_to_decode, loss1);

        readcount = fread(buf_channel2 + COMPRESSED_PARAMETERS, sizeof(float), COMPRESSED_PARAMETERS, file_to_decode);
        readcount = fread(&max_loss_ch2_2,                      sizeof(float), 1                    , file_to_decode);

        read_loss(file_to_decode, loss2);
        
        decode_sample(saved_channel1 + BUFFER_LEN - CROSS_FADE, buf_channel1 + COMPRESSED_PARAMETERS, max_loss_ch1_2, decoder);
        decode_sample(saved_channel2 + BUFFER_LEN - CROSS_FADE, buf_channel2 + COMPRESSED_PARAMETERS, max_loss_ch2_2, decoder);

        for (int i = 0; i < BUFFER_LEN; i++)
        {
            loss1[i] = loss1[i] / DISCRETE_FACTOR_16;
            loss2[i] = loss2[i] / DISCRETE_FACTOR_16;
        }

        for (int i = 0; i < BUFFER_LEN; i++)
        {
            save[2*i]          = saved_channel1[i] - loss1[i];
            save[2*i + 1]      = saved_channel2[i] - loss2[i];
        }

        memcpy(saved_channel1, saved_channel1 + BUFFER_LEN - CROSS_FADE, sizeof(float)*BUFFER_LEN);
        memset(saved_channel1 + BUFFER_LEN, 0, sizeof(float)*(BUFFER_LEN - CROSS_FADE));
        memcpy(saved_channel2, saved_channel2 + BUFFER_LEN - CROSS_FADE, sizeof(float)*BUFFER_LEN);
        memset(saved_channel2 + BUFFER_LEN, 0, sizeof(float)*(BUFFER_LEN - CROSS_FADE));

        sf_writef_float(output, save, BUFFER_LEN);
    }

    sf_close(output);
}