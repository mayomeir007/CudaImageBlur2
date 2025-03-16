#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES
#include "CUDAImageBlur.cuh"
#include <thread>
#include <chrono>

using namespace cv;
using namespace std::chrono;


void HorizontalBlur(Mat& image, Mat& blurredImage, int radius, float sigma)
{
	// Create the horizontal Gaussian kernel
	int kernelSize = 2 * radius + 1;
	std::vector<double> kernel(kernelSize, 0.0);
	double sum = 0.0;

	for (int i = -radius; i <= radius; ++i) {
		double exponent = -(i * i) / (2 * sigma * sigma);
		double value = std::exp(exponent) / (sqrt(2 * M_PI) * sigma);
		kernel[i + radius] = value;
		sum += value;
	}

	// Normalize the kernel
	for (int i = 0; i < kernelSize; ++i) {
		kernel[i] /= sum;
	}

	int depth = image.channels();

	// Apply the horizontal blur pass
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {

			for (int k = -radius; k <= radius; ++k) {
				if (j + k >= 0 && j + k < image.cols)
				{
					blurredImage.data[i * image.cols * depth + j * depth] += unsigned char(kernel[k + radius] * image.data[i * image.cols * depth + (j + k) * depth]);
					if (depth > 1)
					{
						blurredImage.data[i * image.cols * depth + j * depth + 1] += unsigned char(kernel[k + radius] * image.data[i * image.cols * depth + (j + k) * depth + 1]);
						blurredImage.data[i * image.cols * depth + j * depth + 2] += unsigned char(kernel[k + radius] * image.data[i * image.cols * depth + (j + k) * depth + 2]);
					}
				}
			}
		}
	}
}

void VerticalBlur(Mat& image, Mat& blurredImage, int radius, float sigma)
{
	// Create the vertical Gaussian kernel
	int kernelSize = 2 * radius + 1;
	std::vector<double> kernel(kernelSize, 0.0);
	double sum = 0.0;

	for (int i = -radius; i <= radius; ++i) {
		double exponent = -(i * i) / (2 * sigma * sigma);
		double value = std::exp(exponent) / (sqrt(2 * M_PI) * sigma);
		kernel[i + radius] = value;
		sum += value;
	}

	// Normalize the kernel
	for (int i = 0; i < kernelSize; ++i) {
		kernel[i] /= sum;
	}

	int depth = image.channels();

	// Apply the vertical blur pass
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {

			for (int k = -radius; k <= radius; ++k) {
				if (i + k >= 0 && i + k < image.rows)
				{
					blurredImage.data[i * image.cols * depth + j * depth] += float(kernel[k + radius] * image.data[(i + k) * image.cols * depth + j * depth]);
					if (depth > 1)
					{
						blurredImage.data[i * image.cols * depth + j * depth + 1] += float(kernel[k + radius] * image.data[(i + k) * image.cols * depth + j * depth + 1]);
						blurredImage.data[i * image.cols * depth + j * depth + 2] += float(kernel[k + radius] * image.data[(i + k) * image.cols * depth + j * depth + 2]);
					}
				}
			}

		}
	}
}

void blurImageCPU(const char* input, float blurPercent, float* time, Mat* blurredImage)
{
	Mat image;
	image = imread(input, CV_LOAD_IMAGE_COLOR);
	float blurFactor = blurPercent / 100;

	int bradiusHori = int(blurFactor * image.cols / 2);
	Mat temp(image.rows, image.cols, image.type());
	temp.setTo(0);
	int bradiusVerti = int(blurFactor * image.rows / 2);
	//blurredImage = new Mat(image.rows, image.cols, image.type());
	blurredImage->create(image.rows, image.cols, image.type());

	blurredImage->setTo(0);

	auto cpu_start = high_resolution_clock::now();
	HorizontalBlur(image, temp, bradiusHori, bradiusHori * .3f);
	VerticalBlur(temp, *blurredImage, bradiusVerti, bradiusVerti * .3f);
	auto cpu_end = high_resolution_clock::now();
	//namedWindow("CPU Display Window", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	//imshow("CPU Display Window", blurredImage);

	//pass time to 'time' and display message 
	auto duration = duration_cast<microseconds>(cpu_end - cpu_start);
	*time = duration.count() / 1000000.f;
	std::cout << "CPU finished. execution time " << *time << "  seconds." << std::endl;
	//waitKey();
}

__global__ void horizontal_blur(uchar* img, uchar* temp, int depth, int nx, int ny, double* kernel, int radius)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		//printf("x =%d, y=%d \n", ix, iy);

		temp[iy * nx * depth + ix * depth] = 0;
		if (depth > 1)
		{
			temp[iy * nx * depth + ix * depth + 1] = 0;
			temp[iy * nx * depth + ix * depth + 2] = 0;
		}
		for (int k = -radius; k <= radius; ++k) {
			if (ix + k >= 0 && ix + k < nx)
			{
				temp[iy * nx * depth + ix * depth] += unsigned char(kernel[k + radius] * img[iy * nx * depth + (ix + k) * depth]);
				if (depth > 1)
				{
					temp[iy * nx * depth + ix * depth + 1] += unsigned char(kernel[k + radius] * img[iy * nx * depth + (ix + k) * depth + 1]);
					temp[iy * nx * depth + ix * depth + 2] += unsigned char(kernel[k + radius] * img[iy * nx * depth + (ix + k) * depth + 2]);
				}
			}
		}
	}
}

__global__ void vertical_blur(uchar* img, uchar* temp, int depth, int nx, int ny, double* kernel, int radius)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < nx && iy < ny)
	{
		int depth = 3;

		img[iy * nx * depth + ix * depth] = 0;
		if (depth > 1)
		{
			img[iy * nx * depth + ix * depth + 1] = 0;
			img[iy * nx * depth + ix * depth + 2] = 0;
		}
		for (int k = -radius; k <= radius; ++k) {
			if (iy + k >= 0 && iy + k < ny)
			{
				img[iy * nx * depth + ix * depth] += float(kernel[k + radius] * temp[(iy + k) * nx * depth + ix * depth]);
				if (depth > 1)
				{
					img[iy * nx * depth + ix * depth + 1] += float(kernel[k + radius] * temp[(iy + k) * nx * depth + ix * depth + 1]);
					img[iy * nx * depth + ix * depth + 2] += float(kernel[k + radius] * temp[(iy + k) * nx * depth + ix * depth + 2]);
				}

			}
		}
	}
}

double* getGaussianKernel(int radius, float sigma)
{
	// Create the vertical Gaussian kernel
	int kernelSize = 2 * radius + 1;
	double* kernel = new double[kernelSize];

	double* CUDAkernel;
	cudaMalloc((void**)&CUDAkernel, kernelSize * sizeof(double));
	double sum = 0.0;

	for (int i = -radius; i <= radius; ++i) {
		double exponent = -(i * i) / (2 * sigma * sigma);
		double value = std::exp(exponent) / (sqrt(2 * M_PI) * sigma);
		kernel[i + radius] = value;
		sum += value;
	}

	// Normalize the kernel
	for (int i = 0; i < kernelSize; ++i) {
		kernel[i] /= sum;
	}
	cudaMemcpy(CUDAkernel, kernel, kernelSize * sizeof(double), cudaMemcpyHostToDevice);

	delete[] kernel;
	return CUDAkernel;
}

void blurImageGPU(const char* input, float blurPercent, float* time, Mat* blurredImage)
{
	Mat image;
	image = imread(input, CV_LOAD_IMAGE_COLOR);
	float blurFactor = blurPercent / 100;
	//blurredImage->rows = 5;// new Mat(image.rows, image.cols, image.type());
	blurredImage->create(image.rows, image.cols, image.type());

	int bradiusHori = int(blurFactor * image.cols / 2);
	int bradiusVerti = int(blurFactor * image.rows / 2);

	double* kernelHori = getGaussianKernel(bradiusHori, bradiusHori * .3f);
	double* kernelVerti = getGaussianKernel(bradiusVerti, bradiusVerti * .3f);

	int block_x = 128;
	int block_y = 8;
	dim3 blocks(block_x, block_y);
	unsigned int gridDimX = (image.cols + block_x - 1) / block_x;
	unsigned int gridDimY = (image.rows + block_y - 1) / block_y;
	dim3 grid(gridDimX, gridDimY);

	int size = image.cols * image.rows;
	int byte_size = image.channels() * sizeof(uchar) * size;

	uchar* d_img_temp, * d_img_blur;

	cudaMalloc((void**)&d_img_temp, byte_size);
	cudaMalloc((void**)&d_img_blur, byte_size);

	cudaEvent_t start, end;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	cudaMemcpy(d_img_blur, image.data, byte_size, cudaMemcpyHostToDevice);

	horizontal_blur << < grid, blocks >> > (d_img_blur, d_img_temp, image.channels(), image.cols, image.rows, kernelHori, bradiusHori);
	cudaDeviceSynchronize();

	vertical_blur << < grid, blocks >> > (d_img_blur, d_img_temp, image.channels(), image.cols, image.rows, kernelVerti, bradiusVerti);

	cudaDeviceSynchronize();

	//copy back to cpu
	cudaMemcpy(blurredImage->data, d_img_blur, byte_size, cudaMemcpyDeviceToHost);

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time_ms;
	cudaEventElapsedTime(&time_ms, start, end);

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaFree(d_img_temp);
	cudaFree(d_img_blur);
	cudaFree(kernelHori);
	cudaFree(kernelVerti);

	*time = time_ms / 1000.f;

	std::cout << "GPU finished. execution time " << *time << "  seconds." << std::endl;
}

bool CUDAImageBlur(const char* input, float blurPercent)
{
	//set log level of openCV to warning 
	utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_WARNING);

	//allocate in the heap the time reading and Mat for blurred image to be used by the CPU
	float* cpu_time = new float(0);
	Mat* blurredImage_cpu = new Mat;
	//allocate in the heap time reading and Mat for blurred image to be used by the GPU
	float* gpu_time = new float(0);
	Mat* blurredImage_gpu = new Mat;

	std::cout << "The CPU and GPU are ready to race. The are going to apply gaussian blur on image \"" << input << "\", percent: " << blurPercent << std::endl;
	std::cout << "And they're off!" << std::endl;

	std::thread cpu_thread = std::thread(blurImageCPU, input, blurPercent, cpu_time, blurredImage_cpu);

	std::thread gpu_thread = std::thread(blurImageGPU, input, blurPercent, gpu_time, blurredImage_gpu);

	if (gpu_thread.joinable())
	{
		gpu_thread.join();
		namedWindow("GPU Display Window", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("GPU Display Window", *blurredImage_gpu); 
		waitKey(1);
		if (*cpu_time == 0)
		{
			std::cout << "CPU is still running..." << std::endl;
		}
	}
	if (cpu_thread.joinable())
	{
		cpu_thread.join();
		namedWindow("CPU Display Window", WINDOW_NORMAL | WINDOW_KEEPRATIO);
		imshow("CPU Display Window", *blurredImage_cpu);
	}

	std::cout << "GPU performed the image blurr " << *cpu_time / *gpu_time  << " times faster!" << std::endl;

	waitKey();

	//free data in the heap
	delete cpu_time;
	delete gpu_time;
	delete blurredImage_cpu;
	delete blurredImage_gpu;

	return true;
}