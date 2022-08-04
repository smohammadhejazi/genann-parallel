// Libraries
#include "./genann.h"
#include "./genann.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
// Structs
struct nnData
{
    double image[784];
	double label[10];
};
// Consts Variables
char DATASET_PATH[] = "./semeion.data";
double LEARNING_RATE = 0.3;
// Globals
int num_inputs;
int num_hidden_layers;
int num_hidden_layers_neurons;
int num_outputs;
int num_epochs = 50;
int num_dataset_data = 0;
char *dataset_path;
struct nnData* dataset;
// ***********************************************************************
void omp_check() {
	#ifdef _OPENMP
	#else
		printf("[!] OpenMP is off.\n");
		printf("[#] Enable OpenMP.\n");
	#endif
}

double * readData(char* dataPath)
{
	FILE* ptr;
    char ch;

    ptr = fopen(dataPath, "r");
	// count number of lines
	if (NULL == ptr) 
	{
        printf("file can't be opened \n");
		return NULL;
    }
	for (ch = getc(ptr); ch != EOF; ch = getc(ptr))
	{
		if (ch == '\n')
		{
			num_dataset_data++;
		}
	}

	fclose(ptr);

	ptr = fopen(dataPath, "r");
	// read inputs
	dataset = (nnData *)malloc(num_dataset_data * sizeof(struct nnData));
    struct nnData newData;
	int tmp;
	int imageWidth = (int) sqrt(num_inputs);

	for (int line = 0; line < num_dataset_data; line++)
	{
		for (int i = 0; i < imageWidth; i++)
		{
			for (int j = 0; j < imageWidth; j++)
			{
				// fscanf(ptr, "%lf ", &newData.image[i * imageWidth + j]);
				fscanf(ptr, "%d ", &tmp);
				newData.image[i * imageWidth + j] = (double) tmp;
			}
		}
		for (int i = 0; i < num_outputs; i++)
		{
			fscanf(ptr, "%d ", &tmp);
			newData.label[i] = (double) tmp;
		}
		fscanf(ptr, "\n");
		dataset[line] = newData;
	}

    return 0;
}

int main(int argc, char *argv[])
{
	double startTime;
	
	omp_check();
	if (argc == 6 || argc == 7) 
	{
		num_inputs = atoi(argv[1]);
		num_hidden_layers = atoi(argv[2]);
		num_hidden_layers_neurons = atoi(argv[3]);
		num_outputs = atoi(argv[4]);
		dataset_path = argv[5];
		if (argc == 7) {
			num_epochs = atoi(argv[6]);
		}
	}
	else {
		printf("Wrong number of arguments supplied.\n");
		return -1;
	}
	//------------------------------------------------
	// program start point
	//------------------------------------------------
   	startTime = omp_get_wtime();
	// read dataset and save it to global array dataset
	readData(dataset_path);

	// build nn and train
	genann *nn = genann_init(num_inputs, num_hidden_layers, num_hidden_layers_neurons, num_outputs);

	int seventy_percent_index = (int) (num_dataset_data * 0.7);
	for (int epoch = 0; epoch < num_epochs; epoch++)
	{
		printf("Epoch: %d\n", epoch);
		for (int i = 0; i < seventy_percent_index; i++)
		{
			genann_train(nn, dataset[i].image, dataset[i].label, LEARNING_RATE);
		}
	}

	int correct_output = 0;
	for (int i = seventy_percent_index; i < num_dataset_data; i++)
	{
		const double *output = genann_run(nn, dataset[i].image);
		double max_output = -999;
		int max_index = 0;
		// get neural network prediction by getting maximum neuron output
		for (int j = 0; j < num_outputs; j++)
		{
			if (output[j] >= max_output)
			{
				max_output = output[j];
				max_index = j;

			}
		}
		if (dataset[i].label[max_index] == 1)
		{
			correct_output++;
		}
	}

	double accuracy = correct_output / (double) (num_dataset_data - seventy_percent_index) * 100;

	//------------------------------------------------
	// program end point
	//------------------------------------------------
	printf("%0.2f%%\n", accuracy);
	printf("%fs\n", omp_get_wtime() - startTime);

	return 0;
}