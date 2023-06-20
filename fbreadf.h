#include <stdio.h>
#include <stdlib.h>

#define FRAME_LEN 1024

typedef struct _ANSWER{
    double m;
    double delta[FRAME_LEN];
} Answer;

void fbread(FILE* file, float* m, float* delta);
void fbread_answer_version(FILE* file, Answer* ans);