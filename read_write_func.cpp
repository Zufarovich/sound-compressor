#include <stdio.h>
#include <stdlib.h>
#include "read_write_func.hpp"

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