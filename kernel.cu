#include <stdlib.h>
#include <stdio.h>

#include "CUDAImageBlur.cuh"
#include <opencv2/opencv.hpp>


int main(int argc, char** argv)
{
    bool valid = false;
    if (argc > 1)
    {   
        valid = CUDAImageBlur(argv[1], atoi(argv[2]));
    }
    if (!valid)
    {
        printf("Invalid input\n");
    }
    return 0;
}
