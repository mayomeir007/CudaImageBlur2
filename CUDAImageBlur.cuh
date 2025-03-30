#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <opencv2/core/utils/logger.hpp>

//main method. Compares blurring of given image path with CPU and with GPU
bool CUDAImageBlur(const char* input, int blurPercent);

//auxialiary functions
//takes input image 'image' and returns it with horizontal blure pass 'blurredImage'
void HorizontalBlur(const cv::Mat& image, cv::Mat& blurredImage, int radius, float sigma);
//takes input image 'image' and returns it with vertical blure pass 'blurredImage'
void VerticalBlur(const cv::Mat& image, cv::Mat& blurredImage, int radius, float sigma);
//performs CPU gaussian blur on input path to image, saves result on 'blurredImage' and the time it took 'time'
void blurImageCPU(const char* input, int blurPercent, float* time, cv::Mat* blurredImage);


//auxiliary functions for GPU
//kernels
//CUDA kernel performes horizontal blur pass on 'img' using input gassian kernel and and saves in 'temp'
__global__ void horizontal_blur(uchar* img, uchar* temp, int depth, int nx, int ny, const double* kernel, int radius);
//CUDA kernel performes vertical blur pass on 'img' using input gassian kernel and and saves in 'temp'
__global__ void vertical_blur(uchar* img, uchar* temp, int depth, int nx, int ny, const double* kernel, int radius);
//returns gaussina kernal givenn sigma and blur radius and saves in the GPU
double* getGaussianKernel(int radius, float sigma);
//performs GPU gaussian blur on input path to image, saves result on 'blurredImage' and the time it took 'time'
void blurImageGPU(const char* input, int blurPercent, float* time, cv::Mat* blurredImage);


