#define NOMINMAX // so that windows.h does not define min/max macros

#include <algorithm>
#include <iostream>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "../shared/defines.h"
#include "../shared/utils.h"

#define INPUT_DIM 28
#define CONV_MAT_DIM 5
#define CNN_NUM_FEATURES 32
#define MAX_POOL_SIZE 2
#define POOLING_OUTPUT_DIM (INPUT_DIM/MAX_POOL_SIZE)
#define CNN_NUM_WEIGHTS_PER_FEATURE (CONV_MAT_DIM*CONV_MAT_DIM + 1)
#define FC_NUM_WEIGHTS_PER_DIGIT (CNN_NUM_FEATURES * POOLING_OUTPUT_DIM*POOLING_OUTPUT_DIM + 1)
#define NUM_DIGITS 10

using namespace aocl_utils;

// OpenCL Global Variables.
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_kernel cnn_kernel;
cl_kernel fc_kernel;
cl_program program;

cl_uchar *input_images = NULL, *reference_guesses = NULL, *output_guesses = NULL;
cl_short *cnn_weights = NULL, *fc_weights = NULL;
cl_mem input_images_buffer, cnn_weights_buffer, fc_weights_buffer, cnn_out_buffer, output_guesses_buffer;

// Global variables.
std::string imagesFilename;
std::string labelsFilename;
std::string aocxFilename;
std::string deviceInfo;
std::string weightsDir;
int batch_size, batches, n_items;

// Function prototypes.
void classify();
void classify_sw();
void classify_old();
void initCL();
void cleanup();
void teardown(int exit_status = 1);
void print_usage();

int main(int argc, char **argv) {
	// Parsing command line arguments.
	Options options(argc, argv);
	
	// Relative path to images file.
	if(options.has("images")) {
		imagesFilename = options.get<std::string>("images");
	} else {
		imagesFilename = "../../../design_files/t10k-images.idx3-ubyte";
	}
	printf("Using images file \"%s\"\n", imagesFilename.c_str());
	
	// Relative path to labels file.
	if(options.has("labels")) {
		labelsFilename = options.get<std::string>("labels");  
	} else {
		labelsFilename = "../../../design_files/t10k-labels.idx1-ubyte";
	}
	printf("Using labels file \"%s\"\n", labelsFilename.c_str());
	
	// Relative path to aocx filename.
	if(options.has("aocx")) {
		aocxFilename = options.get<std::string>("aocx");  
	} else {
		aocxFilename = "cnn_classifier";
	}
	printf("Using aocx file \"%s.aocx\"\n", aocxFilename.c_str());
	
	// Relative path to weights directory.
	if(options.has("weights_dir")) {
		weightsDir = options.get<std::string>("weights_dir");  
	} else {
		weightsDir = "../../../design_files/weights_fxp";
	}
	printf("Using weights in \"%s\"\n", weightsDir.c_str());
	
	// Read in the images and labels
	n_items = parse_MNIST_images(imagesFilename.c_str(), &input_images);
	if (n_items <= 0){
		printf("ERROR: Failed to parse images file.\n");
		return -1;
	}
	if (n_items != parse_MNIST_labels(labelsFilename.c_str(), &reference_guesses)){
		printf("ERROR: Number of labels does not match number of images\n");
		return -1;
	}
	
	if(options.has("batch_size")) {
		batch_size = options.get<int>("batch_size");
	} else {
		batch_size = 1000;
	}
	if(options.has("batches")) {
		batches = options.get<int>("batches");
	} else {
		batches = 10;
	}
	printf("Classifying %d batches of %d images (total %d images)\n", batches, batch_size, batch_size*batches);
	if (n_items != batch_size*batches) printf("WARNING: opened %d images but will classify %d images\n", n_items, batch_size*batches);
		
	// Allocate some arrays
	output_guesses = (cl_uchar*)alignedMalloc(sizeof(cl_uchar) *batch_size*batches);
	cnn_weights = (cl_short*)alignedMalloc(sizeof(cl_short) * CNN_NUM_FEATURES * CNN_NUM_WEIGHTS_PER_FEATURE);
	fc_weights = (cl_short*)alignedMalloc(sizeof(cl_short) * FC_NUM_WEIGHTS_PER_DIGIT * NUM_DIGITS);
	
	// Read in the weights from the weights files
	char weights_file[256];
	for (unsigned i = 0; i < NUM_DIGITS; i++){
		snprintf(weights_file, 256, "%s/fc_weights_%d", weightsDir.c_str(), i);
		if (!read_int16_weights_file(weights_file, fc_weights+FC_NUM_WEIGHTS_PER_DIGIT*i, FC_NUM_WEIGHTS_PER_DIGIT)){
			printf("ERROR: Failed to read in fc weights\n");
			return -1;
		}
	}
	snprintf(weights_file, 256, "%s/cnn_weights", weightsDir.c_str());
	if (!read_int16_weights_file(weights_file, cnn_weights, CNN_NUM_FEATURES * CNN_NUM_WEIGHTS_PER_FEATURE)){
		printf("ERROR: Failed to read in cnn weights\n");
		return -1;
	}
	
	initCL();
	
	// Start measuring time
	double start = get_wall_time();
	
	classify();
	
	// Stop measuring time.
	double end = get_wall_time();
	printf("TIME ELAPSED: %.2f ms\n", end - start);
	
	int correct = 0;
	for (unsigned i = 0; i < batch_size*batches; i++){
		if (output_guesses[i] == reference_guesses[i]) correct++;
	}
	printf("Predicted %d correctly out of %d (Accuracy: %.2f%%)\n", correct, batch_size*batches, (float)correct*100/(batch_size*batches));
	
	// Teardown OpenCL.
	teardown(0);
}

void classify() {
	size_t size = 1;
	cl_int status;
	int num_inputs = batch_size; 
	cl_event event[2];
	double total_kernel_time = 0;
	
	// Create kernel input and output buffers.
	input_images_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) *INPUT_DIM*INPUT_DIM* batch_size, NULL, &status);
	checkError(status, "Error: could not create input image buffer");
	cnn_weights_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * CNN_NUM_WEIGHTS_PER_FEATURE * CNN_NUM_FEATURES, NULL, &status);
	checkError(status, "Error: could not create cnn weights buffer");
	fc_weights_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * FC_NUM_WEIGHTS_PER_DIGIT * NUM_DIGITS, NULL, &status);
	checkError(status, "Error: could not create fc weights buffer");
	cnn_out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * CNN_NUM_FEATURES * POOLING_OUTPUT_DIM * POOLING_OUTPUT_DIM * batch_size, NULL, &status);
	checkError(status, "Error: could not create output guesses buffer");
	output_guesses_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * batch_size, NULL, &status);
	checkError(status, "Error: could not create output guesses buffer");
	
	// Copy data to weights buffers
	status = clEnqueueWriteBuffer(queue, cnn_weights_buffer, CL_TRUE, 0, sizeof(short) * CNN_NUM_WEIGHTS_PER_FEATURE * CNN_NUM_FEATURES, cnn_weights, 0, NULL, NULL);
	checkError(status, "Error: could not copy data into device");
	status = clEnqueueWriteBuffer(queue, fc_weights_buffer, CL_TRUE, 0, sizeof(short) * FC_NUM_WEIGHTS_PER_DIGIT * NUM_DIGITS, fc_weights, 0, NULL, NULL);
	checkError(status, "Error: could not copy data into device");
	
	// Set arguments to the CNN kernel
	status = clSetKernelArg(cnn_kernel, 0, sizeof(cl_mem), (void*)&input_images_buffer);
	checkError(status, "Error: could not set argument 0");
	status = clSetKernelArg(cnn_kernel, 1, sizeof(cl_mem), (void*)&cnn_weights_buffer);
	checkError(status, "Error: could not set argument 1");
	status = clSetKernelArg(cnn_kernel, 2, sizeof(cl_mem), (void*)&cnn_out_buffer);
	checkError(status, "Error: could not set argument 2");
	status = clSetKernelArg(cnn_kernel, 3, sizeof(int), &num_inputs);
	checkError(status, "Error: could not set argument 3");

	// Set arguments to the FC kernel
	status = clSetKernelArg(fc_kernel, 0, sizeof(cl_mem), (void*)&cnn_out_buffer);
	checkError(status, "Error: could not set argument 0");
	status = clSetKernelArg(fc_kernel, 1, sizeof(cl_mem), (void*)&fc_weights_buffer);
	checkError(status, "Error: could not set argument 1");
	status = clSetKernelArg(fc_kernel, 2, sizeof(cl_mem), (void*)&output_guesses_buffer);
	checkError(status, "Error: could not set argument 2");
	status = clSetKernelArg(fc_kernel, 3, sizeof(int), &num_inputs);
	checkError(status, "Error: could not set argument 3");
	
	// Run the batches
	for (int nth_batch = 0; nth_batch < batches; nth_batch++){
	
		printf("Running batch %d\n",nth_batch);
	
		// Copy data to input image buffer
		status = clEnqueueWriteBuffer(queue, input_images_buffer, CL_TRUE, 0, sizeof(unsigned char) * INPUT_DIM*INPUT_DIM * batch_size, input_images+nth_batch*batch_size*INPUT_DIM*INPUT_DIM, 0, NULL, NULL);
		checkError(status, "Error: could not copy data into device");
		
		// Start measuring time
		double start = get_wall_time();
		
		// Enqueue the CNN kernel. 
		status = clEnqueueTask(queue, cnn_kernel, 0, NULL, &event[0]);
		checkError(status, "Error: failed to enqueue cnn_kernel");
		// Enqueue the FC kernel, but make it wait until CNN kernel is done.
		status = clEnqueueTask(queue, fc_kernel, 1, &event[0], &event[1]);
		checkError(status, "Error: failed to enqueue fc_kernel");
		
		// Wait for command queue to complete pending events.
		status = clFinish(queue);
		checkError(status, "Kernels failed to finish");

		// Stop measuring time.
		double end = get_wall_time();
		total_kernel_time += end - start;
		
		// Read output buffer from kernel.
		status = clEnqueueReadBuffer(queue, output_guesses_buffer, CL_TRUE, 0, sizeof(unsigned char) * batch_size, output_guesses+nth_batch*batch_size, 0, NULL, NULL);
		checkError(status, "Error: could not copy data from device");
	}
	
	printf("KERNEL TIME ELAPSED: %.2f ms\n", total_kernel_time);
	
	clReleaseEvent(event[0]);
	clReleaseEvent(event[1]);
}

void initCL() {
	cl_int status;

	// Start everything at NULL to help identify errors.
	cnn_kernel = NULL;
	fc_kernel = NULL;
	queue = NULL;
	
	// Locate files via. relative paths.
	if(!setCwdToExeDir()) {
		teardown();
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA");
	if (platform == NULL) {
		teardown();
	}

	// Get the first device.
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	checkError (status, "Error: could not query devices");

	char info[256];
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
	deviceInfo = info;

	// Create the context.
	context = clCreateContext(0, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Error: could not create OpenCL context");

	// Create the command queues for the kernels.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile(aocxFilename.c_str(), device);
	std::cout << "Using AOCX: " << binary_file << "\n";
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 1, &device, "", NULL, NULL);
	checkError(status, "Error: could not build program");
	
	// Create the kernels
	cnn_kernel = clCreateKernel(program, "cnn", &status);
	checkError(status, "Failed to create cnn kernel");
	fc_kernel = clCreateKernel(program, "linear_classifier", &status);
	checkError(status, "Failed to create fc kernel");
		
}

void cleanup() {
	// Called from aocl_utils::check_error, so there's an error.
	teardown(-1);
}

void teardown(int exit_status) {
	if(cnn_kernel) clReleaseKernel(cnn_kernel);
	if(fc_kernel) clReleaseKernel(fc_kernel);
	if(queue) clReleaseCommandQueue(queue);
	if(input_images) alignedFree(input_images);
	if(cnn_weights) alignedFree(cnn_weights);
	if(fc_weights) alignedFree(fc_weights);
	if(reference_guesses) alignedFree(reference_guesses);
	if(output_guesses) alignedFree(output_guesses);
	if(input_images_buffer) clReleaseMemObject(input_images_buffer);
	if(cnn_out_buffer) clReleaseMemObject(cnn_out_buffer);
	if(cnn_weights_buffer) clReleaseMemObject(cnn_weights_buffer);
	if(fc_weights_buffer) clReleaseMemObject(fc_weights_buffer);
	if(program) clReleaseProgram(program);
	if(context) clReleaseContext(context);
	
	exit(exit_status);
}

void print_usage() {
	printf("\nUsage:\n");
	printf("\tlinear_classifier [Options] \n\n");
	printf("Options:\n\n");
	printf("--images=<MNIST images file>\n");
	printf("\tThe relative path to the MNIST images file.\n\n");
	printf("--labels=<MNIST labels file>\n");
	printf("\tThe relative path to the MNIST labels file.\n\n");
	printf("--aocx=<AOCX file>\n");
	printf("\tThe relative path to the .aocx file to use.\n\n");
	printf("--weights_dir=<path to weights files>\n");
	printf("\tThe relative path to the weights files to use.\n\n");
	printf("--batch_size=<integer>\n");
	printf("\tThe number of images to classify per batch.\n\n");
	printf("--batches=<integer>\n");
	printf("\tThe number of batches to run.\n\n");
}
