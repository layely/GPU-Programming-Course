
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include <stdio.h>
#include <algorithm>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define BLOCK_WIDTH 32

/*-------------------------------------------*/
/*			Utils
/*-------------------------------------------*/

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

struct CpuTimer {
	int64 start;
	int64 stop;

	CpuTimer()
	{
	}

	void Start()
	{
		start = cv::getTickCount();
	}

	void Stop()
	{
		stop = cv::getTickCount();
	}

	float Elapsed()
	{
		return ((stop - start) / cv::getTickFrequency()) * 1000;
	}
};

// Sum all pixels of an image with depth 3 (3 channels)
double sumImagePixels(cv::Mat input) {
	cv::Scalar sum = cv::sum(input);
	return sum[0] + sum[1] + sum[2];
}

//std::sort(durations.begin(), durations.end());

float average(float data[]) {
	float accum = 0;
	for (int i = 10; i < 90; ++i)
		accum += data[i];

	return accum / 80;
}

float standardDeviation(float data[])
{
	float sum = 0.0, mean, standardDeviation = 0.0;

	int i;

	for (i = 10; i < 90; ++i)
	{
		sum += data[i];
	}

	mean = sum / 80;

	for (i = 0; i < 80; ++i)
		standardDeviation += pow(data[i] - mean, 2);

	return sqrt(standardDeviation / 80);
}

string getFileName(string s, int maskWidth) {
	std::stringstream ss;
	ss << s << maskWidth << ".jpg";
	return ss.str();
}

 
void printLastCudaError() {
	// make the host block until the device is finished with foo
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
	else {
		printf("No CUDA error \n");
	}
}


/*-------------------------------------------*/
/*			CPU "kernel"
/*-------------------------------------------*/

void convolutionCPU(const unsigned char *input, unsigned char *output, int width, int height,
	int step, int channels, const float * mask, int maskWidth)
{
	for (int row = 0; row < height; ++row)
		for (int col = 0; col < width; ++col)
		{
			int rowStart = row - maskWidth / 2;
			int colStart = col - maskWidth / 2;

			for (int currentChannel = 0; currentChannel < channels; ++currentChannel)
			{
				float sum = 0;
				for (int currentMaskRow = 0; currentMaskRow < maskWidth; ++currentMaskRow)
				{
					for (int currentMaskCol = 0; currentMaskCol < maskWidth; ++currentMaskCol)
					{
						int currentRow = rowStart + currentMaskRow;
						int currentCol = colStart + currentMaskCol;

						// Verify we have a valid image pixel
						if (currentRow > -1 && currentRow < height && currentCol > -1 && currentCol < width)
							sum += input[currentRow * step + currentCol * channels + currentChannel] *
							mask[currentMaskRow * maskWidth + currentMaskCol];
					}
				}

				//Make sure pixel values are in the range 0-255 
				if (sum < 0) sum = 0;
				if (sum > 255) sum = 255;

				output[row * step + col * channels + currentChannel] = static_cast<unsigned char>(sum);
			}
		}
}


/*-------------------------------------------*/
/*			GPU (CUDA) kernels
/*-------------------------------------------*/

__global__
void convolutionGPUKernelSharedMem(const unsigned char *input, unsigned char *output, int width, int height,
	int step, int channels, const float * __restrict__ mask, int maskWidth, int tileWidth) 
{
	__shared__ unsigned char Ns[BLOCK_WIDTH][BLOCK_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*tileWidth + ty;
	int col_o = blockIdx.x*tileWidth + tx;
	int row_i = row_o - maskWidth / 2;
	int col_i = col_o - maskWidth / 2;

	// Compute the output value for each channel
	for (int k = 0; k < channels; ++k) {
		// Load image into shared memory.
		// All threads are involved in this operation 
		if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
		{
			Ns[ty][tx] = input[row_i * step + col_i * channels + k];
		}
		else {
			Ns[ty][tx] = 0.0f;
		}

		// Wait for the loading to finish
		__syncthreads();

		// Compute the output value.
		// Not that some threads don't take part into this
		if (ty < tileWidth && tx < tileWidth)
		{
			float sum = 0.0f;

			for (int i = 0; i < maskWidth; i++)
			{
				for (int j = 0; j < maskWidth; j++)
				{
					sum += mask[i * maskWidth + j] * Ns[i + ty][j + tx];
				}
			}

			if (row_o < height && col_o < width) {
				// Normalize output value
				if (sum < 0) sum = 0;
				if (sum > 255) sum = 255;
			
				output[row_o*step + col_o * channels + k] = static_cast<unsigned char>(sum);
			}
		}

		// Wait for computation to finish for this channel 
		// before moving to another channel
		__syncthreads();
	}
}

__global__
void convolutionGPUKernelGlobalMem(const unsigned char *input, unsigned char *output, 
	int width, int height, int step, int channels, const float * mask, int maskWidth)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x; 
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height)
	{
		int rowStart = row - maskWidth / 2;
		int colStart = col - maskWidth / 2;

		for (int currentChannel = 0; currentChannel < channels; ++currentChannel)
		{
			float sum = 0;
			for (int currentMaskRow = 0; currentMaskRow < maskWidth; ++currentMaskRow)
			{
				for (int currentMaskCol = 0; currentMaskCol < maskWidth; ++currentMaskCol)
				{
					int currentRow = rowStart + currentMaskRow;
					int currentCol = colStart + currentMaskCol;

					// Verify we have a valid image pixel
					if (currentRow > -1 && currentRow < height && currentCol > -1 && currentCol < width) 
					{
						sum += input[currentRow * step + currentCol * channels + currentChannel] * 
							   mask[currentMaskRow * maskWidth + currentMaskCol];
					}
				}
			}

			//Make sure pixel values are in the range 0-255
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;

			output[row * step + col * channels + currentChannel] = static_cast<unsigned char>(sum);
		}
	}
}

/*-------------------------------------------*/
/*	Test - Testing kernels defined above
/*-------------------------------------------*/

void convolutionGPU_shared_mem_test(const cv::Mat& input, cv::Mat& output, 
	const float * mask, const int maskWidth)
{
	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step *  output.rows;
	const int maskBytes = sizeof(float) * maskWidth * maskWidth;

	unsigned char *d_output, *d_input;
	float * d_mask;

	const unsigned char * inputMat = input.ptr();
	unsigned char * outputMat = output.ptr();
	int width = input.cols;
	int height = input.rows;
	int step = input.step;
	int channels = input.channels();


	//Specify a reasonable block size
	dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

	//Calculate the tile width which depends on the mask size
	int tileWidth = BLOCK_WIDTH - maskWidth + 1;

	//Calculate grid size to cover the whole image
	const dim3 grid((output.cols - 1) / tileWidth + 1, (output.rows - 1) / tileWidth + 1);

	float durations[100];

	for (int i = 0; i < 100; ++i) {
		GpuTimer timer;

		//Allocate device memory
		cudaMalloc<unsigned char>(&d_input, inputBytes);
		cudaMalloc<unsigned char>(&d_output, outputBytes);
		cudaMalloc<float>(&d_mask, maskBytes);

		//Copy input image to device
		cudaMemcpy(d_input, inputMat, inputBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mask, mask, maskBytes, cudaMemcpyHostToDevice);

		timer.Start();
		convolutionGPUKernelSharedMem << <grid, block >> > (d_input, d_output, width, height,
			step, channels, d_mask, maskWidth, tileWidth);
		timer.Stop();
		//printLastCudaError();

		//Copy input image to device
		cudaMemcpy(outputMat, d_output, outputBytes, cudaMemcpyDeviceToHost);
		
		
		durations[i] = timer.Elapsed();
	}

	std::sort(begin(durations), end(durations));
	printf("CUDA Shared Memory average time: %g\n", average(durations));
	printf("CUDA Shared Memory standard deviation: %g\n", standardDeviation(durations));

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_mask);
}

void convolutionGPU_global_mem_test(const cv::Mat& input, cv::Mat& output, 
	const float * mask, const int maskWidth)
{
	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step *  output.rows;
	const int maskBytes = sizeof(float) * maskWidth * maskWidth;

	unsigned char *d_output, *d_input;
	float * d_mask;

	const unsigned char * inputMat = input.ptr();
	unsigned char * outputMat = output.ptr();
	int width = input.cols;
	int height = input.rows;
	int step = input.step;
	int channels = input.channels();
	
	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, 
		(output.rows + block.y - 1) / block.y);

	float durations[100];

	for (int i = 0; i < 100; ++i) {
		GpuTimer timer;

		
		//Allocate device memory
		cudaMalloc<unsigned char>(&d_input, inputBytes);
		cudaMalloc<unsigned char>(&d_output, outputBytes);
		cudaMalloc<float>(&d_mask, maskBytes);



		//Copy input image to device
		cudaMemcpy(d_input, inputMat, inputBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mask, mask, maskBytes, cudaMemcpyHostToDevice);

		timer.Start();
		convolutionGPUKernelGlobalMem << <grid, block >> > (d_input, d_output, width, height,
			step, channels, d_mask, maskWidth);
		timer.Stop();

		//Copy input image to device
		cudaMemcpy(outputMat, d_output, outputBytes, cudaMemcpyDeviceToHost);
		
		durations[i] = timer.Elapsed();
	}

	std::sort(begin(durations), end(durations));
	printf("CUDA Global Memory average time: %g\n", average(durations));
	printf("CUDA Global Memory standard deviation: %g\n", standardDeviation(durations));

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_mask);
}

void convolutionCPU_test(const cv::Mat& input, cv::Mat& output,const float * mask, int maskWidth)
{
	const unsigned char * inputMat = input.ptr();
	unsigned char * outputMat = output.ptr();
	int width = input.cols;
	int height = input.rows;
	int step = input.step;
	int channels = input.channels();

	float durations[100];

	for (int i = 0; i < 100; ++i) {
		CpuTimer timer;
		timer.Start();
		convolutionCPU(inputMat, outputMat, width, height, step, channels, mask, maskWidth);
		timer.Stop();
		durations[i] = timer.Elapsed();
	}

	std::sort(begin(durations), end(durations));
	printf("CPU average time: %g\n", average(durations));
	printf("CPU standard deviation: %g\n", standardDeviation(durations));

	namedWindow("OutputCPU", 1);
	imshow("OutputCPU", output);
}

int main()
{
	string inputPath = "GiantLobster.jpg";

	printf("Init\n");

	//Load window;
	Mat input = cv::imread(inputPath, CV_LOAD_IMAGE_COLOR);

	const float mask3x3[] = { -1, 0, 1,
							  -2, 0, 2, 
							  -1, 0, 1 };
	const float mask5x5[] = { -1, -2, 0, 2, 1, 
							  -4, -8, 0, 8, 4, 
							  -6, -12, 0, 12, 6, 
							  -4, -8, 0, 8, 4, 
							  -1, -2, 0, 2, 1 };
	const float mask7x7[] = { 1, 1, 1, 0, -1, -1, -1,
							  1, 2, 2, 0, -2, -2, -1,
							  1, 2, 3, 0, -3, -2, -1,
							  1, 2, 3, 0, -3, -2, -1,
							  1, 2, 3, 0, -3, -2, -1,
							  1, 2, 2, 0, -2, -2, -1,
							  1, 1, 1, 0, -1, -1, -1 };

	Size newSize(input.size().width, input.size().height);
	Mat outputCPU(newSize, input.type());
	Mat outputGPUGlobalMem(newSize, input.type());
	Mat outputGPUSharedMem(newSize, input.type());

	const float * mask = mask3x3;
	const int maskWidth = 3;

	printf("\n ---Running benchmarking. Please wait...---\n");

	printf("Mask width: %d \n", maskWidth);
	
	printf("** Running on CPU...\n");
	convolutionCPU_test(input, outputCPU, mask, maskWidth);

	printf("** Running on GPU: global memory\n");
	convolutionGPU_global_mem_test(input, outputGPUGlobalMem, mask, maskWidth);
	
	printf("** Running on GPU: shared memory\n");
	convolutionGPU_shared_mem_test(input, outputGPUSharedMem, mask, maskWidth);

	printf("\n ---Benchmarking Done---\n");

	//--
	// Show original and resulting images

	namedWindow("Original Image", 1);
	imshow("Original Image", input);

	namedWindow("Convolution CPU", 1);
	imshow("Convolution CPU", outputCPU);
	imwrite(getFileName("./Convolution_CPU.jpg", maskWidth), outputCPU);

	namedWindow("Convoluton CUDA Global Memory", 1);
	imshow("Convoluton CUDA Global Memory", outputGPUGlobalMem);
	imwrite(getFileName("./Convolution_CUDA_Global_Memory.jpg", maskWidth), outputGPUGlobalMem);
	//imwrite(outputPath, output);

	namedWindow("Convoluton CUDA Shared Memory", 1);
	imshow("Convoluton CUDA Shared Memory", outputGPUSharedMem);
	std::stringstream ss;
	imwrite(getFileName("./Convolution_CUDA_Shared_Memory", maskWidth), outputGPUGlobalMem);

	//--
	// Print resulting images sum 

	double sumCPUGlobalMem = sumImagePixels(outputCPU - outputGPUGlobalMem);
	printf("Resulting sum (cpu - cuda_global_mem): %g\n", sumCPUGlobalMem);

	double sumSharedMem = sumImagePixels(outputCPU - outputGPUSharedMem);
	printf("Resulting sum (cpu - cuda_shared_mem): %g\n", sumSharedMem);

	//wait for the user to press any key:
	waitKey(0);

	return 0;
}