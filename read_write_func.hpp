#include <sndfile.h>

#define MAX_CHANNELS 6
#define ERROR_OPEN_INPUT -1
#define ERROR_OPEN_OUTPUT -2
#define EXTRA_CHANNELS -3

int   open_sf_write(char* file, SNDFILE** sf_file, SF_INFO* sf_file_info);

/**
* open_sf_write
* @breif Opens sound file for writing and sets all essential parameters(format, channels etc.)
* @param char* file - path to file to write
* @param SNDFILE** sf_file - pointer to pointer for sndfile variable to write
* @param SF_INFO* sf_file_info - pointer to structure with information about file to write
*/

int   open_sf_read(char* file, SNDFILE** sf_file, SF_INFO* sf_file_info);

/**
* open_sf_read
* @breif Opens sound file for reading
* @param char* file - path to file to read
* @param SNDFILE** sf_file - pointer to pointer for sndfile variable to read
* @param SF_INFO* sf_file_info - pointer to structure with information about file to read
*/