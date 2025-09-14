#define INPUT_DIM 28
#define CONV_MAT_DIM 5
#define CNN_NUM_FEATURES 32
#define MAX_POOL_SIZE 2
#define POOLING_OUTPUT_DIM (INPUT_DIM/MAX_POOL_SIZE)
#define CNN_NUM_WEIGHTS_PER_FEATURE (CONV_MAT_DIM*CONV_MAT_DIM + 1)

// CNN kernel. Take square (INPUT_DIMxINPUT_DIM) input image, apply CNN_NUM_FEATURES conv matrices, 
// do POOLING_OUTPUT_DIMxPOOLING_OUTPUT_DIM max pooling, and write the output to features_out. 
// Takes weights array of CNN_NUM_FEATURES*CNN_NUM_WEIGHTS_PER_FEATURE weights. 
// The first CNN_NUM_WEIGHTS_PER_FEATURE-1 weights of each set of weights correspond to 
// the convolution matrix, and last weight is the bias.
__kernel
void cnn(	global const unsigned char * restrict frame_in, 
					global const short * restrict weights, 
					global int * restrict features_out,
					const int num_images)
{
	int pixel = 0;
	int feature = 0;
	int weight = 0;
	short local_weights[CNN_NUM_FEATURES][CNN_NUM_WEIGHTS_PER_FEATURE];
	
	// Cache the weights
	for (int c = 0; c < CNN_NUM_FEATURES; c++)
		for (int w = 0; w < CNN_NUM_WEIGHTS_PER_FEATURE; w++) 
			local_weights[c][w] = weights[weight++];
	
	
	for (int image = 0; image < num_images; image++){
		
		unsigned char frame[INPUT_DIM][INPUT_DIM]; 
		
		// load and pad input image
		for (int y = 0; y < INPUT_DIM; y++){
			for (int x = 0; x < INPUT_DIM; x++){
				frame[y][x] = frame_in[pixel++];
			}
		}
		
		for (int c = 0; c < CNN_NUM_FEATURES; c++){
			// Iterate over 2x2 blocks of the input image (to do the 2x2 pooling)
			for (int y_block = 0; y_block < POOLING_OUTPUT_DIM; y_block++){
				for (int x_block = 0; x_block < POOLING_OUTPUT_DIM; x_block++){
					
					// The max value for the max pool block of pixels
					int max = 0;
					
					// iterate the pool block of pixels
					for (int y_off = 0; y_off < MAX_POOL_SIZE; y_off++){
						for (int x_off = 0; x_off < MAX_POOL_SIZE; x_off++){
							
							int y = y_block*MAX_POOL_SIZE + y_off;
							int x = x_block*MAX_POOL_SIZE + x_off;
							
							// local_weights[CONV_MAT_DIM*CONV_MAT_DIM] is the bias value
							int feature_accum = local_weights[c][CONV_MAT_DIM*CONV_MAT_DIM];
							#pragma unroll
							for (int cy = 0; cy < CONV_MAT_DIM; cy++){
								#pragma unroll
								for (int cx = 0; cx < CONV_MAT_DIM; cx++){
									if (y+cy >= CONV_MAT_DIM/2 && y+cy < INPUT_DIM+CONV_MAT_DIM/2 &&
										x+cx >= CONV_MAT_DIM/2 && x+cx < INPUT_DIM+CONV_MAT_DIM/2)
										feature_accum += local_weights[c][cy*CONV_MAT_DIM + cx]*frame[y+cy-CONV_MAT_DIM/2][x+cx-CONV_MAT_DIM/2];
								}
							}
							
							if (feature_accum > max) max = feature_accum;
						}
					}
					
					features_out[feature++] = max;
				}
			}
		}
	}
}

#define ARRAY_DIM (CNN_NUM_FEATURES*POOLING_OUTPUT_DIM*POOLING_OUTPUT_DIM)
#define FC_NUM_WEIGHTS_PER_DIGIT (CNN_NUM_FEATURES * POOLING_OUTPUT_DIM*POOLING_OUTPUT_DIM + 1)
#define NUM_DIGITS 10

__kernel void linear_classifier(global const int * restrict features, 
								global const short * restrict weights,
								global unsigned char * restrict guesses,
								const int num_images)
{
	unsigned char guess = 0;
	int score[NUM_DIGITS] = {0};
	
	short local_weights[FC_NUM_WEIGHTS_PER_DIGIT*NUM_DIGITS];
	
	for (int i = 0; i < FC_NUM_WEIGHTS_PER_DIGIT*NUM_DIGITS; i++){
		local_weights[i] = weights[i];
	}
	
	for (int n = 0; n < num_images; n++){
		
		guess = 0;
		
		#pragma unroll
		for (int i = 0; i < NUM_DIGITS; i++)
			score[i] = local_weights[i*FC_NUM_WEIGHTS_PER_DIGIT + FC_NUM_WEIGHTS_PER_DIGIT-1];
		
		#pragma unroll 2
		for (int x = 0; x < ARRAY_DIM; x++){
			#pragma unroll
			for (int i = 0; i < NUM_DIGITS; i++){
				score[i] += features[n*ARRAY_DIM + x]*local_weights[i*FC_NUM_WEIGHTS_PER_DIGIT+x];
			}
		}
	
		// Determine highest score
		#pragma unroll
		for (int i = 1; i < NUM_DIGITS; i++){
			if (score[i] > score[guess]) guess = i;
		}
		guesses[n] = guess;
	}
}
