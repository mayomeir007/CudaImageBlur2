#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <opencv2/core/utils/logger.hpp>

//main method. Compares blurring of given image path with CPU and with GPU
bool CUDAImageBlur(const char* input, float blurPercent);

//auxialiary functions
void HorizontalBlur(cv::Mat& image, cv::Mat& blurredImage, int radius, float sigma);

void VerticalBlur(cv::Mat& image, cv::Mat& blurredImage, int radius, float sigma);

void blurImageCPU(const char* input, float blurPercent, float* time, cv::Mat* blurredImage);


//aucualiry functions for GPU
//kernels
__global__ void horizontal_blur(uchar* img, uchar* temp, int depth, int nx, int ny, double* kernel, int radius);

__global__ void vertical_blur(uchar* img, uchar* temp, int depth, int nx, int ny, double* kernel, int radius);

double* getGaussianKernel(int radius, float sigma);

void blurImageGPU(const char* input, float blurPercent, float* time, cv::Mat* blurredImage);


