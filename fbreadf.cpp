#include <stdio.h>
#include <stdlib.h>

#define FRAME_LEN 1024

typedef struct _ANSWER{
    float m;
    float delta[FRAME_LEN];
} Answer;

void fbread_answer_version(FILE* file, Answer* ans){
    float m = 1;
    int po2 = 0;
    fread(&m, sizeof(float), 1, file);
    fread(&po2, sizeof(int), 1, file);
    printf("pow_of_2:%d\n", po2);
    char buffer = 0;
    int count = 0;
    int position = 0;
    ans->m = m;

    while(count<FRAME_LEN){
        bool readOne = false;
        bool readSing = false;
        bool readP = false;
        int num = 0;
        while (true){
            int getbit = 0;
            int bit = position%8;
            int byte = position/8;
            position++;
            if (!bit)
                fread(&buffer, sizeof(char), 1, file);
            getbit = buffer>>(7-bit) & 1;

            if (!getbit && !readOne){
                ans->delta[count] = 0;
                break;
            }else{
                if(!readOne){
                    readOne = true;
                    continue;
                }else if(!readSing){
                    ans->delta[count] = getbit ? 1:-1;
                    readSing = true;
                    continue;
                }else if(!readP){
                    num += getbit;
                    if (!getbit){
                        ans->delta[count]*= po2*num;
                        num = 0;
                        readP = true;
                    }
                    continue;
                }else{
                    for(int p = po2; p>0; p/=2)
                        num = num<<1+getbit;
                    ans->delta[count]=(ans->delta[count]+num)*m;
                    break;
                }
            }
        }
        count++;
    }
}


void fbread(FILE* file, float* scale, float* delta){
    Answer ans;
    fbread_answer_version(file, &ans);
    *scale = ans.m;
    for(int i = 0; i<FRAME_LEN; i++)
        delta[i] = ans.delta[i];
}

/*
int main(){
    FILE* file = fopen("1.f1", "r");
    Answer ans;
    float buf[FRAME_LEN];
    float m;
    
    fbread(file, buf, &m);
    //fbread_answer_version(file, &ans);
    fclose(file);
    printf("%lf\n", m);
    for(int i = 0; i<10; i++)
        printf("%lf\n", buf[i]);
    
    
}
*/

