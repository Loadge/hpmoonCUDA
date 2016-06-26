/**
 * @file evaluation.cu
 * @author Miguel Sánchez Tello
 * @date 26/06/2016
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
#include <cub-1.5.2/cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>

/********************************* Methods ********************************/

const int BLOCK_SIZE=128;

/**
 * @brief Given a integer, returns it's next power of two.
 * @param n The initial number.
 * @return Next power of two integer of n.
 */
static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;

    return ++n;
}

/**
 * @brief Part of K-means algorithm that is made within the GPU that minimizes the within-cluster sum of squares.
 * @param bigdistCentroids The instances that are considered the center of each cluster. They are formatted to make appropiate calculations.
 * @param NextPowerTotalDistances The size of bigdistCentroids.
 * @param BlockSumWithin Where the results of each concurrent block are stored for further processing in CPU.
 * @param newMapping The mapping the algorithm is going to build in this iteration.
 * @param newMapping The count of instances that each cluster contains.
 */
__global__ static 
void cuda_WithinCluster(
							const int * NextPowerTotalDistances,
							const float * __restrict__ bigdistCentroids,
							float * __restrict__ BlockSumWithin
							)
{
	const int gpu_NextPowerTotalDistances = *(NextPowerTotalDistances);
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Specialize BlockReduce for type float
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceT;

	//Allocat temporary storage in shared memory
	__shared__ typename BlockReduceT::TempStorage temp_storage_float;

	float result;
	if(idx < gpu_NextPowerTotalDistances)
		result = BlockReduceT(temp_storage_float).Sum(bigdistCentroids[idx]);
	__syncthreads();
	if(threadIdx.x == 0){
		BlockSumWithin[blockIdx.x] = result;
	}
}

/**
 * @brief Part of K-means algorithm that is made within the GPU that calculates the euclidean distances between each instance.
 * @param dataBase The database which will contain the instances and the features.
 * @param centroids The instances that are considered the center of each cluster.
 * @param member_chromosome The chromosome necessary to calculate each euclidean distance.
 * @param newMapping The mapping the algorithm is going to build in this iteration.
 * @param newMapping The count of instances that each cluster contains.
 */
__global__ static
void cuda_Convergence_Euclidean(	
						float *dataBase, 
						float *centroids,
						unsigned char *member_chromosome,
						float * distCentroids,
						bool * newMapping,
						int * samples_in_k
					)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int d_totalDistances = KMEANS * N_INSTANCES;

	__shared__ int sharedThreadLater[N_INSTANCES];

	for(int i=threadIdx.x; i < d_totalDistances; i+= blockDim.x){
		newMapping[i] = false;
	}
	
	for(int i=threadIdx.x; i < KMEANS; i+= blockDim.x){
		samples_in_k[i] = 0;
	}
	__syncthreads();
	for (int i = idx; i < N_INSTANCES; i += blockDim.x) {
/* -- */
		float minDist = INFINITY;
		int selectCentroid = -1;
		int pos = N_FEATURES * i;

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
			distCentroids[posDistCentr + i] = euclidean; //Access to global memory. OPT
			if (euclidean < minDist) {
				minDist = euclidean;
				selectCentroid = k;
			}
		}//k
/* -- */
		sharedThreadLater[i]= selectCentroid;
		newMapping[(selectCentroid * N_INSTANCES) + i] = true;
		__syncthreads();
	}//i

	//We keep this in GPU to avoid unnecessary memory tranfers
	__syncthreads();
	if(idx==0){
		for(int i=0; i<N_INSTANCES; i++){
			samples_in_k[sharedThreadLater[i]]++;
		}
	}
}

/**
 * @brief K-means algorithm implemented with C-CUDA which minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS). Uses CPU and GPU for maximum performance.
 * @param pop Current population.
 * @param begin The first individual to evaluate.
 * @param end The "end-1" position is the last individual to evaluate.
 * @param selInstances The instances chosen as initial centroids.
 * @param dataBase The database which will contain the instances and the features.
 */
void CUDA_kmeans(individual *pop, const int begin, const int end, const int *const selInstances, const float *const dataBase) {

	bool *mapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));
	bool *newMapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));

	const int totalDistances = KMEANS * N_INSTANCES;
	const int totalCoord = KMEANS * N_FEATURES;
	
	int nextPowerTotalDistances = nextPowerOfTwo(totalDistances+1);

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

	bool * d_auxMapping;
	checkCudaErrors(cudaMalloc((void **)&d_auxMapping, size_5));		

	int * d_samples_in_k; 
	size_t size_6 = KMEANS * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&d_samples_in_k, size_6));	

	int * d_posCentroids;
	size_t size_7 = sizeof(int);
	checkCudaErrors(cudaMalloc(&d_posCentroids, size_7));

	int * d_posDistCentr;
	checkCudaErrors(cudaMalloc(&d_posDistCentr, size_7));

	int * d_NextPowerTotalDistances;
	checkCudaErrors(cudaMalloc(&d_NextPowerTotalDistances, size_7));

	bool * d_bigmapping;		
	size_t size_8 = nextPowerTotalDistances * sizeof(bool);
	checkCudaErrors(cudaMalloc((void **)&d_bigmapping, size_8));

	bool * d_bigNewMapping;
	checkCudaErrors(cudaMalloc((void **)&d_bigNewMapping, size_8));

	float * d_bigdistCentroids;	
	size_t size_9 = nextPowerTotalDistances * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&d_bigdistCentroids, size_9));

	//Decide number of blocks and threads for each parallel section.
	cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

	unsigned int numEuclideanThreadsPerBlock = 128;
	unsigned int numEuclideanBlocks = ((N_INSTANCES+numEuclideanThreadsPerBlock)-1) / numEuclideanThreadsPerBlock;
	if(numEuclideanBlocks==0){numEuclideanBlocks=1;}

	//In order for CUBLAS to support reduction, numWithinThreadsPerBlock must be a power of two.
	unsigned int numWithinThreadsPerBlock = BLOCK_SIZE;
	unsigned int numWithinBlocks = ((totalDistances+totalDistances)-1) / BLOCK_SIZE;
	if(numWithinBlocks==0){numWithinBlocks=1;}
	if(numEuclideanBlocks > (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount)){
		printf("WARNING: Your CUDA hardware has insufficient blocks!.\n");
		numEuclideanBlocks = (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
	}

	/*	----------------------------------TODO:----------------------------------
			 calculate the amount of global memory necessary for the program to execute. Use this:
			    char msg[256];
    			SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
            			(float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
			    printf("%s", msg);
    	----------------------------------TODO:----------------------------------
    */

	// Evaluate all the individuals
	for (int ind = begin; ind < end; ++ind) {
		float centroids[KMEANS * N_FEATURES];

		// The centroids will have the selected features of the individual
		for (int k = 0; k < KMEANS; ++k) {
			int posDataBase = selInstances[k] * N_FEATURES;
			int posCentr = k * N_FEATURES;

			for (int f = 0; f < N_FEATURES; ++f) {
				if (pop[ind].chromosome[f] & 1) {
					centroids[posCentr + f] = dataBase[posDataBase + f];
				}
			}
		}

		//Capture individual chromosome for the GPU
		unsigned char h_member_chromosome[N_FEATURES];
		for(int i=0; i<N_FEATURES; i++){
			h_member_chromosome[i] = pop[ind].chromosome[i];
		}

		//Allocate extra device necessary structures 
		float * d_BlockSumWithin;
		size_t size_12 = numWithinBlocks * sizeof(float);
		checkCudaErrors(cudaMalloc((void **)&d_BlockSumWithin, size_12));

		int * d_bigMappings;
		size_t size_13 = nextPowerTotalDistances * sizeof(int);
		checkCudaErrors(cudaMalloc((void **)&d_bigMappings, size_13));

		checkCudaErrors(cudaMemcpy(d_dataBase, dataBase, size_1, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_centroids, centroids, size_2, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_member_chromosome, h_member_chromosome, size_3, cudaMemcpyHostToDevice));

		bool gpu_converged = false;
		float gpu_distCentroids[KMEANS * N_INSTANCES];
		bool gpu_mapping[KMEANS * N_INSTANCES];
		bool gpu_newMapping[KMEANS * N_INSTANCES];
		int gpu_samples_in_k[KMEANS];
		float gpu_centroids[KMEANS * N_FEATURES];
		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !gpu_converged ; ++maxIter) {//52
			//Use GPU for the heavy computing part.
			cuda_Convergence_Euclidean <<< numEuclideanBlocks, numEuclideanThreadsPerBlock >>> (	
																		d_dataBase, 
																		d_centroids, 
																		d_member_chromosome, 
																		d_distCentroids, 
																		d_newMapping,
																		d_samples_in_k
																		);
			cudaDeviceSynchronize();
/* -- */
			checkCudaErrors(cudaMemcpy(gpu_distCentroids,d_distCentroids, size_4, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_newMapping, 	 d_newMapping, size_5, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_mapping, 	 d_mapping, size_5, cudaMemcpyDeviceToHost));

			//Has the algorithm converged? We use CPU.
			gpu_converged = true;
			int sum=0;
			for(int i=0; i<totalDistances; i++){
				sum+= (gpu_mapping[i] != gpu_newMapping[i]) ? 0 : 1;
			}
			if(sum != totalDistances ){
						gpu_converged = false;
			}

			if(!gpu_converged){
				checkCudaErrors(cudaMemcpy(gpu_centroids, d_centroids, size_2, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(gpu_newMapping, d_newMapping, size_5, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(gpu_samples_in_k, d_samples_in_k, size_6, cudaMemcpyDeviceToHost));
				
				// Update the position of the centroids using CPU
				for (int k = 0; k < KMEANS; ++k) {
					int posCentroids = k * N_FEATURES;
					int posMapping = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						float sum = 0.0f;
						if (pop[ind].chromosome[f] & 1) {
							for (int i = 0; i < N_INSTANCES; ++i) {
								if (gpu_newMapping[posMapping + i]) {
									sum += dataBase[(N_FEATURES * i) + f];
								}
							}
							if(gpu_samples_in_k[k] == 0){
								gpu_centroids[posCentroids + f] = 0;
							}else{
								gpu_centroids[posCentroids + f] = sum / gpu_samples_in_k[k];
							}
						}//if chromosome
					}//for nfeatures
				}//for KMEANS

				// Swap GPU mapping tables 
				checkCudaErrors(cudaMemcpy(d_mapping, 		gpu_newMapping, size_5, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(d_newMapping, 	gpu_mapping, size_5, 	cudaMemcpyHostToDevice));
				//New centroids, thanks to CPU work
				checkCudaErrors(cudaMemcpy(d_centroids, 	gpu_centroids, size_2, 	cudaMemcpyHostToDevice));
			}//!converged
		}//maxIter	52

		/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) *************/
/* -- */
		//In order for CUBLAS to support reduction, size of the structure used must match the number of threads we are using.
		//numWithinThreadsPerBlock is a power of two.
		float gpu_bigdistCentroids[nextPowerTotalDistances];

		int k=0;
		for (; k < totalDistances; ++k) {  
				gpu_bigdistCentroids[k] = gpu_distCentroids[k] * gpu_mapping[k];			
		}
		for(; k < nextPowerTotalDistances; k++){
			gpu_bigdistCentroids[k] = 0.0f;
		}
		checkCudaErrors(cudaMemcpy(d_bigdistCentroids,	gpu_bigdistCentroids, 	size_9, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_NextPowerTotalDistances, &nextPowerTotalDistances, size_7, cudaMemcpyHostToDevice));

		// Within-cluster
		cuda_WithinCluster <<< numWithinBlocks, numWithinThreadsPerBlock>>> (
														d_NextPowerTotalDistances, 
														d_bigdistCentroids,
														d_BlockSumWithin
														);
		cudaDeviceSynchronize();

		float gpu_SumWithin_2=0;
		float gpu_BlockSumWithin[numWithinBlocks];

		checkCudaErrors(cudaMemcpy(gpu_BlockSumWithin, 	 d_BlockSumWithin, 		size_12, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(gpu_mapping,		 	 d_mapping, 			size_5,  cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(gpu_distCentroids, d_distCentroids	, 		size_4,  cudaMemcpyDeviceToHost));

		for(int i=0; i < numWithinBlocks; i++){
			gpu_SumWithin_2 += gpu_BlockSumWithin[i];
		}

		// Inter-cluster  //TODO: paralelize this
		float gpu_SumInter=0.0f;
		for (int k = 0; k < KMEANS; ++k) {
			int gpu_posCentroids = k * N_FEATURES;
			for (int i = gpu_posCentroids + N_FEATURES; i < totalCoord; i += N_FEATURES) {
				float sum = 0.0f;
				for (int f = 0; f < N_FEATURES; ++f) {
					if (pop[ind].chromosome[f] & 1) {
						sum += (gpu_centroids[gpu_posCentroids + f] - gpu_centroids[i + f]) * (gpu_centroids[gpu_posCentroids + f] - gpu_centroids[i + f]);
					}
				}
				gpu_SumInter += sqrt(sum);
			}
		}

		// First objective function (Within-cluster sum of squares (WCSS))
		pop[ind].fitness[0] = gpu_SumWithin_2;

		// Second objective function (Inter-cluster sum of squares (ICSS))
		pop[ind].fitness[1] = gpu_SumInter;

		// Third objective function (Number of selected features)
		//pop[ind].fitness[2] = (float) nSelFeatures;
/* -- */
		checkCudaErrors(cudaFree(d_BlockSumWithin));
/* -- */
		//WCSS and ICSS minimization process
	}//for each individual
/* -- */
	// Resources used are released
	checkCudaErrors(cudaFree(d_dataBase));
	checkCudaErrors(cudaFree(d_centroids));
	checkCudaErrors(cudaFree(d_member_chromosome));
	checkCudaErrors(cudaFree(d_distCentroids));
	checkCudaErrors(cudaFree(d_mapping));
	checkCudaErrors(cudaFree(d_newMapping));
	checkCudaErrors(cudaFree(d_auxMapping));
	checkCudaErrors(cudaFree(d_samples_in_k));
	checkCudaErrors(cudaFree(d_posCentroids));
	checkCudaErrors(cudaFree(d_posDistCentr));
	checkCudaErrors(cudaFree(d_NextPowerTotalDistances));
	checkCudaErrors(cudaFree(d_bigmapping));
	checkCudaErrors(cudaFree(d_bigdistCentroids));
/* -- */
	free(mapping);
	free(newMapping);

}//CUDA_kmeans

/**
 * @brief Sequential K-means algorithm which minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS).
 * @param pop Current population.
 * @param begin The first individual to evaluate.
 * @param end The "end-1" position is the last individual to evaluate.
 * @param selInstances The instances choosen as initial centroids.
 * @param dataBase The database which will contain the instances and the features.
 */
void kmeans(individual *pop, const int begin, const int end, const int *const selInstances, const float *const dataBase) {

	bool *mapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));
	bool *newMapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));

	// Evaluate all the individuals
	for (int ind = begin; ind < end; ++ind) {
		const int totalCoord = KMEANS * N_FEATURES;
		float centroids[KMEANS * N_FEATURES];

		// The centroids will have the selected features of the individual
		for (int k = 0; k < KMEANS; ++k) {
			int posDataBase = selInstances[k] * N_FEATURES;
			int posCentr = k * N_FEATURES;

			for (int f = 0; f < N_FEATURES; ++f) {
				if (pop[ind].chromosome[f] & 1) {
					centroids[posCentr + f] = dataBase[posDataBase + f];
				}
			}
		}



		/******************** Convergence process *********************/

		// Initialize the array of minimum distances and the mapping table
		const int totalDist = KMEANS * N_INSTANCES;
		float distCentroids[KMEANS * N_INSTANCES];
		int samples_in_k[KMEANS];

		// Initialize the mapping table
		for (int i = 0; i < totalDist; ++i) {
			mapping[i] = false;
		}

		// To avoid poor performance, at most "MAX_ITER_KMEANS" iterations are executed
		bool converged = false;
		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !converged; ++maxIter) {

			// The mapping table is cleaned in each iteration
			for (int i = 0; i < totalDist; ++i) {
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
					int posCentr = k * N_FEATURES;
					int posDistCentr = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						if (pop[ind].chromosome[f] & 1) {
							sum += (dataBase[pos + f] - centroids[posCentr + f]) * (dataBase[pos + f] - centroids[posCentr + f]);
						}
					}
					

					float euclidean = sqrt(sum);
					distCentroids[posDistCentr + i] = euclidean;
					if (euclidean < minDist) {
						minDist = euclidean;
						selectCentroid = k;
					}
				}

				newMapping[(selectCentroid * N_INSTANCES) + i] = true;
				samples_in_k[selectCentroid]++;
			}

			// Has the algorithm converged?
			converged = true;
			for (int k = 0; k < KMEANS && converged; ++k) { 
				int posMap = k * N_INSTANCES;
				for (int i = 0; i < N_INSTANCES && converged; ++i) {
					if (newMapping[posMap + i] != mapping[posMap + i]) {
						converged = false;
					}
				}
			}

			if (!converged) {

				// Update the position of the centroids
				for (int k = 0; k < KMEANS; ++k) {
					int posCentr = k * N_FEATURES;
					int posMap = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						float sum = 0.0f;
						if (pop[ind].chromosome[f] & 1) {
							for (int i = 0; i < N_INSTANCES; ++i) {
								if (newMapping[posMap + i]) {
									sum += dataBase[(N_FEATURES * i) + f];
								}
							}
							centroids[posCentr + f] = sum / samples_in_k[k];
						}
					}
				}

				// Swap mapping tables
				bool *aux = newMapping;
				newMapping = mapping;
				mapping = aux;
			}
	}


		/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) *************/

		float sumWithin = 0.0f;
		float sumInter = 0.0f;
		for (int k = 0; k < KMEANS; ++k) {
			int posCentr = k * N_FEATURES;
			int posDistCentr = k * N_INSTANCES;

			// Within-cluster
			for (int i = 0; i < N_INSTANCES; ++i) {
				if (mapping[posDistCentr + i]) {
					sumWithin += distCentroids[posDistCentr + i];
				}
			}

			// Inter-cluster
			for (int i = posCentr + N_FEATURES; i < totalCoord; i += N_FEATURES) {
				float sum = 0.0f;
				for (int f = 0; f < N_FEATURES; ++f) {
					if (pop[ind].chromosome[f] & 1) {
						sum += (centroids[posCentr + f] - centroids[i + f]) * (centroids[posCentr + f] - centroids[i + f]);
					}
				}
				sumInter += sqrt(sum);
			}
		}

		// First objective function (Within-cluster sum of squares (WCSS))
		pop[ind].fitness[0] = sumWithin;

		// Second objective function (Inter-cluster sum of squares (ICSS))
		pop[ind].fitness[1] = sumInter;

		// Third objective function (Number of selected features)
		//pop[ind].fitness[2] = (float) nSelFeatures;
	}

	// Resources used are released
	free(mapping);
	free(newMapping);
}

/**
 * @brief Evaluation of each individual.
 * @param pop Current population.
 * @param begin The first individual to evaluate.
 * @param end The "end-1" position is the last individual to evaluate.
 * @param dataBase The database which will contain the instances and the features.
 * @param nInstances The number of instances (rows) of the database.
 * @param nFeatures The number of features (columns) of the database.
 * @param nObjectives The number of objectives.
 * @param selInstances The instances chosen as initial centroids.
 */
void evaluation(individual *pop, const int begin, const int end, const float *const dataBase, const int nInstances, const int nFeatures, const unsigned char nObjectives, const int *const selInstances) {


	/************ Kmeans algorithm ***********/

	// Evaluate all the individuals and get the first and second objective for them
	kmeans(pop, begin, end, selInstances, dataBase);


	/******************** Fitness normalization *********************/

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
 * @brief Evaluation of each individual.
 * @param pop Current population.
 * @param begin The first individual to evaluate.
 * @param end The "end-1" position is the last individual to evaluate.
 * @param dataBase The database which will contain the instances and the features.
 * @param nInstances The number of instances (rows) of the database.
 * @param nFeatures The number of features (columns) of the database.
 * @param nObjectives The number of objectives.
 * @param selInstances The instances chosen as initial centroids.
 */
void CUDA_evaluation(individual *pop, const int begin, const int end, const float *const dataBase, const int nInstances, const int nFeatures, const unsigned char nObjectives, const int *const selInstances) {


	/************ Kmeans algorithm ***********/

	// Evaluate all the individuals and get the first and second objective for them
	CUDA_kmeans(pop, begin, end, selInstances, dataBase);


	/******************** Fitness normalization *********************/

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
 * @brief Gets the hypervolume measure of the population.
 * @param pop Current population.
 * @param nIndFront0 The number of individuals in the front 0.
 * @param nObjectives The number of objectives.
 * @param referencePoint The necessary reference point for calculation.
 * @return The value of the hypervolume.
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

/**
 * @brief CUDA Kernel used to generate a random number per thread
 * @param d_out The random numbers generated
 * @param max The upper limit of the generation interval
 * @param min The bottom limit of the generation interval
 */
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
 * @brief Gets the initial centroids (instances chosen randomly).
 * @param selInstances Where the instances chosen as initial centroids will be stored.
 * @param nInstances The number of instances (rows) of the database.
 */
void getCentroids(int *selInstances, const int nInstances) {
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
}


/**
 * @brief Generates gnuplot code for data display.
 * @param dataName The name of the file which will contain the fitness of the individuals in the first Pareto front.
 * @param plotName The name of the file which will contain the gnuplot code for data display.
 * @param imageName The name of the file which will contain the image with the data (graphic).
 * @param pop Current population.
 * @param nIndFront0 The number of individuals in the front 0.
 * @param nObjectives The number of objectives.
 * @param referencePoint The reference point used for the hypervolume calculation.
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