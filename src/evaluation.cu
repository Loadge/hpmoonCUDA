/**
 * @file evaluation.cpp
 * @author Juan José Escobar Pérez
 * @date 20/06/2015
 * @brief File with the necessary implementation for the evaluation of the individuals
 *
 */

/********************************** Includes **********************************/

#include "evaluation.h"
#include "hv.h"
#include <stdio.h> // fprintf...
#include <stdlib.h> // malloc, rand...
#include <string.h> // memset...
#include <math.h> // sqrt, INFINITY...

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>

/********************************* Methods ********************************/

static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

__global__ static
void cuda_Euclidean(	float *dataBase, 
						float *centroids,
						unsigned char *member_chromosome,	
						float * distCentroids,				//
						bool * d_mapping,					//common
						bool * newMapping,					//common
						int *samples_in_k  				//cada hebra tiene uno
					)
{
	extern __shared__ int sharedMemory[]; //main OPT point

	const int d_totalDistances = KMEANS * N_INSTANCES;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	bool converged = false;
		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !converged; ++maxIter) {
			// The mapping table is cleaned in each iteration
			if(idx < d_totalDistances){			//Works when number of threads >=  KMEANS * N_INSTANCES
				newMapping[idx] = false;
			}
			if(idx < KMEANS){
				samples_in_k[idx] = 0;
			}

//------------------------------SEGUIR TOCANDO ESTO-------------------------------------------
			//A single thread executes the necessary work to compute an instance. Future improvements can me made. OPT
			if(idx < N_INSTANCES){
				float minDist = INFINITY;
				int selectCentroid = -1;
				int pos = N_FEATURES * idx;

				for (int k = 0; k < KMEANS; ++k) {
					float sum = 0.0f;
					int posCentroids = k * N_FEATURES;
					int posDistCentr = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						if (member_chromosome[f] & 1) {
							//Multiple accesses to global memory. Better if they were in shared memory. OPT
							sum += (dataBase[pos + f] - centroids[posCentroids + f]) * (dataBase[pos + f] - centroids[posCentroids + f]);
						}
					}//f

					float euclidean = sqrt(sum);
					distCentroids[posDistCentr + idx] = euclidean; //Access to global memory. OPT
					if (euclidean < minDist) {
						minDist = euclidean;
						selectCentroid = k;
					}
				}//k

				newMapping[(selectCentroid * N_INSTANCES) + idx] = true;
				samples_in_k[selectCentroid]++;
				__syncthreads();
			}

			__syncthreads();
		}
}


/**
 * @brief K-means algorithm which minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS)
 * @param pop Current population
 * @param begin The first individual to evaluate
 * @param end The "end-1" position is the last individual to evaluate
 * @param selInstances The instances chosen as initial centroids
 * @param dataBase The database which will contain the instances and the features
 */
void kmeans(individual *pop, const int begin, const int end, const int *const selInstances, const float *const dataBase) {

	bool *mapping = 	(bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));
	bool *newMapping = 	(bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));

	// Evaluate all the individuals
	for (int ind = begin; ind < end; ++ind) {
		const int totalCoord = KMEANS * N_FEATURES;	//Nuevo para cada individuo
		float centroids[KMEANS * N_FEATURES];		//Nuevo para cada individuo
		// The centroids will have the selected features of the individual
		for (int k = 0; k < KMEANS; ++k) {
			int posDataBase = selInstances[k] * N_FEATURES;
			int posCentroids = k * N_FEATURES;

			for (int f = 0; f < N_FEATURES; ++f) {
				if (pop[ind].chromosome[f] & 1) {
					centroids[posCentroids + f] = dataBase[posDataBase + f];
				}
			}
		}



		/******************** Convergence process *********************/

		// Initialize the array of minimum distances and the mapping table
		const int totalDistances = KMEANS * N_INSTANCES;	//Number of times we calculate a distance
		float distCentroids[KMEANS * N_INSTANCES];	//Distance of each instance to each centroid
		int samples_in_k[KMEANS]; 					//Number of samples in each sector, formed by each centroid

		// Initialize the mapping table
		for (int i = 0; i < totalDistances; ++i) {
			mapping[i] = false;
		}







		/* ZONA PARALELA*/

		cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

//        printf("\n%d procesadores!", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

		//Allocate device memory-------------------------------------------
		float *d_dataBase;
		size_t size_1 = N_INSTANCES * N_FEATURES * sizeof(float);
		checkCudaErrors(cudaMalloc((void **)&d_dataBase, size_1));

		float *d_centroids;
		size_t size_2 = KMEANS * N_FEATURES * sizeof(float);
		checkCudaErrors(cudaMalloc((void **)&d_centroids, size_2));

		unsigned char *d_member_chromosome;
		size_t size_3 = N_FEATURES * sizeof(unsigned char);
		checkCudaErrors(cudaMalloc((void **)&d_member_chromosome, size_3));

		float * d_distCentroids;
		size_t size_4 = KMEANS * N_INSTANCES * sizeof(float);
		checkCudaErrors(cudaMalloc((void **)&d_distCentroids, size_4));

		bool * d_mapping;
		size_t size_5 = KMEANS * N_INSTANCES * sizeof(bool);
		checkCudaErrors(cudaMalloc((void **)&d_mapping, size_5));

		bool * d_newMapping;
		checkCudaErrors(cudaMalloc((void **)&d_newMapping, size_5));

		int *d_samples_in_k;
		size_t size_6 = KMEANS * sizeof(int);
		checkCudaErrors(cudaMalloc((void **)&d_samples_in_k, size_6));


		//Capture individual chromosome
		unsigned char h_member_chromosome[N_FEATURES];
		for(int i=0; i<N_FEATURES; i++){
			h_member_chromosome[i] = pop[ind].chromosome[i];
		}


		//Se decide el nº de bloques
		const unsigned int numThreadsPerBlock = nextPowerOfTwo(KMEANS * N_INSTANCES);
		const unsigned int numBlocks = (numThreadsPerBlock)/ (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		const unsigned int BlockSharedDataSize = N_INSTANCES * N_FEATURES * sizeof(float) +
												 KMEANS * N_FEATURES * sizeof(float) +
												 N_FEATURES * sizeof(unsigned char) +
												 KMEANS * N_INSTANCES * sizeof(float) +
												 KMEANS * N_INSTANCES * sizeof(bool) +
												 KMEANS * N_INSTANCES * sizeof(bool) +
												 KMEANS * sizeof(int);
		if (BlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        	printf("WARNING: Your CUDA hardware has insufficient block shared memory.\n");
    	}

		//Copy values in device memory-------------------------------------
		checkCudaErrors(cudaMemcpy(d_dataBase, dataBase, size_1, cudaMemcpyHostToDevice));   //common to block
		checkCudaErrors(cudaMemcpy(d_centroids, centroids, size_2, cudaMemcpyHostToDevice)); //common 
		checkCudaErrors(cudaMemcpy(d_member_chromosome, h_member_chromosome, size_3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_distCentroids, distCentroids, size_4, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_mapping, mapping, size_5, cudaMemcpyHostToDevice));
		//Values for this variables are initialized within the device kernel
		//checkCudaErrors(cudaMemcpy(d_newMapping, newMapping, size_5, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(d_samples_in_k, samples_in_k, size_6, cudaMemcpyHostToDevice));

		cuda_Euclidean <<< numBlocks, numThreadsPerBlock, BlockSharedDataSize >>> (	d_dataBase, 
																					d_centroids, 
																					d_member_chromosome, 
																					d_distCentroids, 
																					d_mapping, 
																					d_newMapping, 
																					d_samples_in_k);

		printf("\nnumThreadsPerBlock: %d", numThreadsPerBlock);
		printf("\nnumBlocks: %d", numBlocks);
		printf("\n Numerajo: %d", (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount));
		printf("\n Se van a lanzar %d hebras repartidas en %d bloques", numThreadsPerBlock*numBlocks, numBlocks);

		// To avoid poor performance, up to "MAX_ITER_KMEANS" iterations are executed
		bool converged = false;
		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !converged; ++maxIter) {

			// The mapping table is cleaned in each iteration
			for (int i = 0; i < totalDistances; ++i) {
				newMapping[i] = false;
			}
			for (int i = 0; i < KMEANS; ++i) {
				samples_in_k[i] = 0;
			}




			// Calculate all distances (Euclidean distance) between each instance and the centroids
			for (int i = 0; i < N_INSTANCES; ++i) {
				float minDist = INFINITY;
				int selectCentroid = -1;
				int pos = N_FEATURES * i;
				for (int k = 0; k < KMEANS; ++k) {
					float sum = 0.0f;
					int posCentroids = k * N_FEATURES;
					int posDistCentr = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						if (pop[ind].chromosome[f] & 1) {
							sum += (dataBase[pos + f] - centroids[posCentroids + f]) * (dataBase[pos + f] - centroids[posCentroids + f]);
						}
					}//f

					float euclidean = sqrt(sum);
					distCentroids[posDistCentr + i] = euclidean;
					if (euclidean < minDist) {
						minDist = euclidean;
						selectCentroid = k;
					}
				}//k

				newMapping[(selectCentroid * N_INSTANCES) + i] = true;
				samples_in_k[selectCentroid]++;
			}//i

			// Has the algorithm converged?
			converged = true;
			for (int k = 0; k < KMEANS && converged; ++k) { 
				int posMapping = k * N_INSTANCES;
				for (int i = 0; i < N_INSTANCES && converged; ++i) {
					if (newMapping[posMapping + i] != mapping[posMapping + i]) {
						converged = false;
					}
				}
			}

			if (!converged) {

				// Update the position of the centroids
				for (int k = 0; k < KMEANS; ++k) {
					int posCentroids = k * N_FEATURES;
					int posMapping = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						float sum = 0.0f;
						if (pop[ind].chromosome[f] & 1) {
							for (int i = 0; i < N_INSTANCES; ++i) {
								if (newMapping[posMapping + i]) {
									sum += dataBase[(N_FEATURES * i) + f];
								}
							}
							centroids[posCentroids + f] = sum / samples_in_k[k];
						}
					}
				}

				// Swap mapping tables
				bool *aux = newMapping;
				newMapping = mapping;
				mapping = aux;
			}
		}//convergence process
		/* ZONA PARALELA*/
		checkCudaErrors(cudaFree(d_dataBase));   //common to block
		checkCudaErrors(cudaFree(d_centroids)); //common 
		checkCudaErrors(cudaFree(d_member_chromosome));
		checkCudaErrors(cudaFree(d_distCentroids));
		checkCudaErrors(cudaFree(d_mapping));
		checkCudaErrors(cudaFree(d_newMapping));
		checkCudaErrors(cudaFree(d_samples_in_k));

		/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) ************* /

		float sumWithin = 0.0f;
		float sumInter = 0.0f;
		for (int k = 0; k < KMEANS; ++k) {
			int posCentroids = k * N_FEATURES;
			int posDistCentr = k * N_INSTANCES;

			// Within-cluster
			for (int i = 0; i < N_INSTANCES; ++i) {
				if (mapping[posDistCentr + i]) {
					sumWithin += distCentroids[posDistCentr + i];
				}
			}

			// Inter-cluster
			for (int i = posCentroids + N_FEATURES; i < totalCoord; i += N_FEATURES) {
				float sum = 0.0f;
				for (int f = 0; f < N_FEATURES; ++f) {
					if (pop[ind].chromosome[f] & 1) {
						sum += (centroids[posCentroids + f] - centroids[i + f]) * (centroids[posCentroids + f] - centroids[i + f]);
					}
				}
				sumInter += sqrt(sum);
			}
		}//WCSS and ICSS minimization process

		// First objective function (Within-cluster sum of squares (WCSS))
		pop[ind].fitness[0] = sumWithin;

		// Second objective function (Inter-cluster sum of squares (ICSS))
		pop[ind].fitness[1] = sumInter;

		// Third objective function (Number of selected features)
		//pop[ind].fitness[2] = (float) nSelFeatures;
/* -- */
	}//for each individual

	// Resources used are released
	free(mapping);
	free(newMapping);

}//kmeans




/**
 * @brief Evaluation of each individual
 * @param pop Current population
 * @param begin The first individual to evaluate
 * @param end The "end-1" position is the last individual to evaluate
 * @param dataBase The database which will contain the instances and the features
 * @param nInstances The number of instances (rows) of the database
 * @param nFeatures The number of features (columns) of the database
 * @param nObjectives The number of objectives
 * @param selInstances The instances chosen as initial centroids
 */
void evaluation(individual *pop, const int begin, const int end, const float *const dataBase, const int nInstances, const int nFeatures, const unsigned char nObjectives, const int *const selInstances) {


	/************ Kmeans algorithm ***********/

	// Evaluate all the individuals and get the first and second objective for them
	kmeans(pop, begin, end, selInstances, dataBase);


	/******************** Fitness normalization ********************* /

	int totalInd = end - begin;
	for (unsigned char obj = 0; obj < nObjectives; ++obj) {

		// Fitness vector average
		float average = 0;
		for (int i = begin; i < end; ++i) {
			average += pop[i].fitness[obj];
		}

		average /= totalInd;

		// Fitness vector variance
		float variance = 0;
		for (int i = begin; i < end; ++i) {
			variance += (pop[i].fitness[obj] - average) * (pop[i].fitness[obj] - average);
		}
		variance /= (totalInd - 1);

		// Fitness vector standard deviation
		float std_deviation = sqrt(variance);

		// The second objective is a maximization problem. x_new must be negative
		if (obj == 1) {

			// Normalize a set of continuous values using SoftMax (based on the logistic function)
			for (int i = begin; i < end; ++i) {
				float x_scaled = (pop[i].fitness[obj] - average) / std_deviation;
				float x_new = 1.0f / (1.0f + exp(-x_scaled));
				pop[i].fitness[obj] = -x_new;
			}
		}
		else {

			// Normalize a set of continuous values using SoftMax (based on the logistic function)
			for (int i = begin; i < end; ++i) {
				float x_scaled = (pop[i].fitness[obj] - average) / std_deviation;
				float x_new = 1.0f / (1.0f + exp(-x_scaled));
				pop[i].fitness[obj] = x_new;
			}
		}
	}
	/* -- */
}


/**
 * @brief Gets the hypervolume measure of the population
 * @param pop Current population
 * @param nIndFront0 The number of individuals in the front 0
 * @param nObjectives The number of objectives
 * @param referencePoint The necessary reference point for calculation
 * @return The value of the hypervolume
 */
float getHypervolume(const individual *const pop, const int nIndFront0, const unsigned char nObjectives, const double *const referencePoint) {

	// Generation the points for the calculation of the hypervolume
	double *points = new double[nObjectives * nIndFront0];
	for (int i = 0; i < nIndFront0; ++i) {
		for (unsigned char obj = 0; obj < nObjectives; ++obj) {
			points[(i * nObjectives) + obj] = pop[i].fitness[obj];
		}
	}

	float hypervolume = fpli_hv(points, nObjectives, nIndFront0, referencePoint);
	delete[] points;

	return hypervolume;
}


__global__ void cudaRand(int *d_out, const int max, const int min)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);

    float my_rand_float = curand_uniform_double(&state);
    my_rand_float *= (max - min+0.99999);
    my_rand_float += min;
    int my_rand = (int)truncf(my_rand_float);

//    assert(my_rand <= max);
//    assert(my_rand >= min);

    d_out[i] = my_rand;
}


/**
 * @brief Gets the initial centroids (instances chosen randomly)
 * @param selInstances Where the instances chosen as initial centroids will be stored
 * @param nInstances The number of instances (rows) of the database
 */
void getCentroids(int *selInstances, const int nInstances) {

/* -- * /
										 
										//OPTIMIZAR <<<<<---- El número de bloques
										//ARREGLAR  <<<<<---- numeros no repetidos
	int * h_v = new int[KMEANS];
	int * d_out;
	//1D grid of 1D blocks
	cudaMalloc((void**)&d_out, KMEANS*sizeof(int));
	cudaRand<<<1, KMEANS >>> (d_out, 0, KMEANS);
	cudaMemcpy(h_v, d_out, KMEANS * sizeof(int), cudaMemcpyDeviceToHost);
	

	for(size_t i=0; i<KMEANS;i++){
		printf("%d ", h_v[i]);
	}
	selInstances = h_v;
	
	printf("\n");
	cudaFree(d_out);
	delete[] h_v;
/* -- */
	// The init centroids will be instances chosen randomly (Forgy's Method)
	for (int k = 0; k < KMEANS; ++k) {
		bool exists = false;
		bool alternate = false;
		int randomInstance;

		/* -- */
		// Avoid repeat centroids
		do {
			randomInstance = rand() % nInstances;
			if(alternate){randomInstance++;if(randomInstance==nInstances){randomInstance=randomInstance-2;}}
			exists = false;

			// Look if the generated index already exists
			for (int kk = 0; kk < k && !exists; ++kk) {
				exists = (randomInstance == selInstances[kk]);
				if(alternate){
					alternate = false;
				}else{
					alternate = true;
				}
			}
		} while (exists);
		/* -- */

		selInstances[k] = randomInstance;
	}
	printf("\nCentroides: ");
	for(int i=0;i<KMEANS;i++){
		printf("%d ", selInstances[i]);
	}
	printf("\n");
}


/**
 * @brief Generates gnuplot code for data display
 * @param dataName The name of the file which will contain the fitness of the individuals in the first Pareto front
 * @param plotName The name of the file which will contain the gnuplot code for data display
 * @param imageName The name of the file which will contain the image with the data (graphic)
 * @param pop Current population
 * @param nIndFront0 The number of individuals in the front 0
 * @param nObjectives The number of objectives
 * @param referencePoint The reference point used for the hypervolume calculation
 */
void generateGnuplot(const char *dataName, const char *plotName, const char *imageName, const individual *const pop, const int nIndFront0, const unsigned char nObjectives, const double *const referencePoint) {

	// Open the data file
	FILE *f_data;
	f_data = fopen(dataName, "w");
	if (!f_data) {
		fprintf(stderr, "Error: An error ocurred opening or writting the data file\n");
		exit(-1);
	}

	// Write the data
	fprintf(f_data, "#Objective0");
	for (unsigned char obj = 1; obj < nObjectives; ++obj) {
		fprintf(f_data, "\tObjective%d", obj);
	}
	for (int i = 0; i < nIndFront0; ++i) {
		fprintf(f_data, "\n%f", pop[i].fitness[0]);
		for (unsigned char obj = 1; obj < nObjectives; ++obj) {
			fprintf(f_data, "\t%f", pop[i].fitness[obj]);
		}
	}
	fclose(f_data);

	// Gnuplot is only available for two objectives
	if (nObjectives == 2) {

		// Open the gnuplot script file
		FILE *f_plot;
		f_plot = fopen(plotName, "w");
		if (!f_data) {
			fprintf(stderr, "Error: An error ocurred opening or writting the plot file\n");
			exit(-1);
		}

		// Write the code
		fprintf(f_plot, "#!/usr/bin/gnuplot\n");
		fprintf(f_plot, "set terminal png size 1024,600\n");
		fprintf(f_plot, "set output '%s.png'\n", imageName);
		fprintf(f_plot, "set multiplot\n");
		fprintf(f_plot, "set xlabel \"Objective 0\"\n");
		fprintf(f_plot, "set grid\n");
		fprintf(f_plot, "set title \"Pareto front\"\n");
		fprintf(f_plot, "set ylabel \"Objective 1\"\n");
		fprintf(f_plot, "set size 0.9,0.9\n");
		fprintf(f_plot, "set origin 0.00,0.05\n");
		fprintf(f_plot, "set key center top\n");
		fprintf(f_plot, "plot [0:1][-1:1] '< sort %s' using 1:%d title \"Front 0\" with lp,\\\n", dataName, nObjectives);
		fprintf(f_plot, "\t\"<echo '%f %f'\" title \"Reference point\" with points,\\\n", referencePoint[0], referencePoint[1]);
		fprintf(f_plot, "\t0 title \"Top pareto limit\" with lp;\n");
		fprintf(f_plot, "set nomultiplot\n");
		fprintf(f_plot, "reset\n");
		fclose(f_plot);
	}
	else {
		fprintf(stdout, "Gnuplot is only available for two objectives. Not generated gnuplot file\n");
	}
}