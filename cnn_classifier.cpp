#include <stdio.h>
#include "../shared/defines.h"
#include "../shared/utils.h"

#define CONV_MAT_DIM 5
#define CNN1_FEATURES 32
#define CNN1_INPUT_DIM 28
#define MAX_POOL_SIZE 2
#define POOL1_OUTPUT_DIM (CNN1_INPUT_DIM/MAX_POOL_SIZE)
#define NN_OUTPUT_SIZE (POOL1_OUTPUT_DIM*POOL1_OUTPUT_DIM*CNN1_FEATURES)
#define CONV_MAT_WEIGHTS (CONV_MAT_DIM*CONV_MAT_DIM+1)

unsigned char *images = NULL; 
unsigned char *labels = NULL;
short *FC_W = NULL;
short *CNN_W = NULL;

void cleanup();

int fc(int cnn_output[][POOL1_OUTPUT_DIM][POOL1_OUTPUT_DIM], short* W){
	int neuron = 0;
	int k = 0;
	
	for (int c = 0; c < CNN1_FEATURES; c++){
		for (int y = 0; y < POOL1_OUTPUT_DIM; y++){
			for (int x = 0; x < POOL1_OUTPUT_DIM; x++){
				neuron += W[k++] * (cnn_output[c][y][x]);
			}
		}
	}
	
	// Add the bias
	return neuron + W[k];
}

void classify(const char* images_file, const char* labels_file){
	bool status = true;
	char weights_file[256];
	int n_correct = 0;
	int n_items = parse_MNIST_images(images_file, &images);
	int items_tested = 0;
	int pixel = 0;
	
	if (n_items <= 0){
		printf("ERROR: Failed to parse images file.\n");
		status = false;
	}
	if (status && n_items != parse_MNIST_labels(labels_file, &labels)){
		printf("ERROR: Number of labels does not match number of images\n");
		status = false;
	}
	
	// NN_OUTPUT_SIZE+1 (1 for the bias)
	if (status) FC_W = (short*) malloc(NUM_DIGITS * (NN_OUTPUT_SIZE+1) * sizeof(short));
	if (status) CNN_W = (short*) malloc(CNN1_FEATURES * CONV_MAT_WEIGHTS * sizeof(short));
		
	// Read in the fc weights
	for (int i = 0; i < NUM_DIGITS && status; i++){
		snprintf(weights_file, 256, "../../design_files/weights_fxp/fc_weights_%d", i);
		status = status && read_int16_weights_file(weights_file, FC_W+(NN_OUTPUT_SIZE+1)*i, NN_OUTPUT_SIZE+1);
	}
	
	// Read in the cnn weights
	snprintf(weights_file, 256, "../../design_files/weights_fxp/cnn_weights");
	status = status && read_int16_weights_file(weights_file, CNN_W, CONV_MAT_WEIGHTS*CNN1_FEATURES);
	
	printf("Starting Predictions on %d items\n",n_items);
	
	// Start measuring classification time
	double start = get_wall_time();
	
	// Predict the digits on the test set
	for (int i = 0; i < n_items && status; i++){
		unsigned char frame[CNN1_INPUT_DIM][CNN1_INPUT_DIM]; 
		int cnn_output[CNN1_FEATURES][POOL1_OUTPUT_DIM][POOL1_OUTPUT_DIM]; 
		
		// Load in the frame 
		for (int y = 0; y < CNN1_INPUT_DIM; y++){
			for (int x = 0; x < CNN1_INPUT_DIM; x++){
				frame[y][x] = images[pixel++];
			}
		}
		
		// First CNN layer
		for (int c = 0; c < CNN1_FEATURES; c++){
			
			for (int y = 0; y < CNN1_INPUT_DIM; y++){
				for (int x = 0; x < CNN1_INPUT_DIM; x++){
					// CNN_W[c*CONV_MAT_WEIGHTS + 25] is the bias value
					int feature_accum = CNN_W[c*CONV_MAT_WEIGHTS + 25];
					for (int cy = 0; cy < CONV_MAT_DIM; cy++){
						for (int cx = 0; cx < CONV_MAT_DIM; cx++){
							if (y+cy >= CONV_MAT_DIM/2 && y+cy < CNN1_INPUT_DIM+CONV_MAT_DIM/2 &&
								x+cx >= CONV_MAT_DIM/2 && x+cx < CNN1_INPUT_DIM+CONV_MAT_DIM/2)
								feature_accum += CNN_W[c*CONV_MAT_WEIGHTS + cy*CONV_MAT_DIM + cx]*frame[y+cy-CONV_MAT_DIM/2][x+cx-CONV_MAT_DIM/2];
						}
					}
					
					// max pool
					if ((x&(MAX_POOL_SIZE-1)) == 0 && (y&(MAX_POOL_SIZE-1)) == 0) // top left pixel
						cnn_output[c][y/MAX_POOL_SIZE][x/MAX_POOL_SIZE] = feature_accum > 0 ? feature_accum : 0;
					else // other 3 pixels
						cnn_output[c][y/MAX_POOL_SIZE][x/MAX_POOL_SIZE] = feature_accum > cnn_output[c][y/MAX_POOL_SIZE][x/MAX_POOL_SIZE] ? feature_accum : cnn_output[c][y/MAX_POOL_SIZE][x/MAX_POOL_SIZE];
				}
			}
		}
		
		int max_score = VERY_NEGATIVE_NUMBER;
		unsigned char current_digit_guess = 0;
		for (int j = 0; j < NUM_DIGITS; j++){
			int score = fc(cnn_output, FC_W + (NN_OUTPUT_SIZE+1)*j);
			if (score > max_score){
				current_digit_guess = j;
				max_score = score;
			}
		}
		if (current_digit_guess == labels[i]) n_correct++;
		items_tested++;
	}
	
	// Stop measuring the filter time.
	double end = get_wall_time();
	printf("TIME ELAPSED: %.2f ms\n", end - start);
	printf("Predicted %d correctly out of %d (Accuracy: %.2f%%)\n", n_correct, items_tested, (float)n_correct * 100 / (float)items_tested);
	
	return;
}

int main(int argc, char *argv[]) {
	
	classify("../../design_files/t10k-images.idx3-ubyte", "../../design_files/t10k-labels.idx1-ubyte");
	
	cleanup();
	
	return 0;
}

void cleanup() {
	if (FC_W) free(FC_W);
	if (CNN_W) free(CNN_W);
	if (images) free(images);
	if (labels) free(labels);
}
