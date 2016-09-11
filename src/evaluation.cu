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

const int BLOCK_SIZE=1024;
const int BLOCK_SIZE_2=1024;
const int N = 1 << 20;

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
 * @brief 
 * @param 
 * @param 
 * @param 
 * @param 
 * @param 
 */
__global__ static 
void cuda_Convergence_Samples(
							int * ind,
							int * samples_in_k,
							int * newSamples
							)
{
	const int individual = *ind;

	const int tamSamples = KMEANS;

	int posIndSamples = individual * tamSamples;
	for(int i=0; i<N_INSTANCES; i++){
		samples_in_k[ posIndSamples + newSamples[i] ]++;
	}
}

/**
 * @brief 
 * @param 
 * @param 
 * @param 
 * @param 
 * @param 
 */
__global__ static 
void cuda_Convergence_SwapMappings(		//49
							int * ind,
							bool * popConverged,
							bool * mapping,
							bool * newMapping
							)
{
	const int individual = *ind;
	if(!popConverged[individual]){
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const int tamMapping = KMEANS * N_INSTANCES;
		int posIndMapping = individual * tamMapping;

		__shared__ int auxMapping[tamMapping];

		for (int i = idx; i < tamMapping; i += blockDim.x) {
			auxMapping[i]					= newMapping[posIndMapping + i];
			newMapping[posIndMapping + i]	= mapping[posIndMapping + i];
			mapping[posIndMapping + i]		= auxMapping[i];
		}
	}//if
}

/**
 * @brief 
 * @param 
 * @param 
 * @param 
 * @param 
 * @param 
 */
__global__ static 
void cuda_Convergence_Update(		//49
							int * ind,
							bool * popConverged,
							float *centroids,
							unsigned char *member_chromosomes,
							int * samples_in_k,
							bool * newMapping,
							float *dataBase
							)
{
	const int individual = *ind;
	if(!popConverged[individual]){
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		const int tamSamples = 		KMEANS;
		const int tamCentroids = 	KMEANS * N_FEATURES;
		const int tamMapping = 		KMEANS * N_INSTANCES;

		int posIndMapping = 	individual * tamMapping;
		int posIndChromosome = 	individual * N_FEATURES;
		int posIndSamples = 	individual * tamSamples;
		int posIndCentroids = 	individual * tamCentroids;

		// Update the position of the centroids
		for (int k = 0; k < KMEANS; ++k) {
		//for (int k = idx; k < KMEANS; k += blockDim.x) {
			int posCentroids = (k * N_FEATURES);
			int posMap = k * N_INSTANCES;
			//for (int f = 0; f < N_FEATURES; ++f) {
			for (int f = idx; f < N_FEATURES; f += blockDim.x) {
				float sum = 1.0f;
//				if (member_chromosomes[posIndChromosome + f] & 1) {
					for (int j = 0; j < N_INSTANCES; ++j) {
						if (newMapping[posIndMapping + posMap + j]) {
							sum += dataBase[(N_FEATURES * j) + f];
//							sum += 1.0f;
						}
					}
					if(samples_in_k[posIndSamples + k] == 0){
						centroids[posIndCentroids + posCentroids + f] = 0;
					}else{
						centroids[posIndCentroids + posCentroids + f] = sum / samples_in_k[posIndSamples + k];
					}
//				}//if chromosome
			}//for nfeatures
		}//for KMEANS
	}//if
}



/**
 * @brief 
 * @param 
 * @param 
 * @param 
 * @param 
 * @param 
 */
__global__ static 
void cuda_Convergence_Check(		//50
							int * ind,
							bool * mapping,
							bool * newMapping,
							bool * popConverged
							)
{
	const int individual = *ind;
	if(!popConverged[individual]){

		const int tamMapping = KMEANS * N_INSTANCES;
		int posIndMapping = *ind * tamMapping;
		const int totalDistances = KMEANS * N_INSTANCES;

		//int idx = blockIdx.x * blockDim.x + threadIdx.x;
		//Has the algorithm converged for this individual?	//TODO: use shared memory insted of global
															//TODO: use more than 1 thread (reduction for 'sum') like the one in cuda_WithinCluster
		popConverged[individual] = true;	//53 GPU
		int sum=0;
		for(int i=0; i<totalDistances; i++){
			sum+= (mapping[posIndMapping + i] != newMapping[posIndMapping + i]) ? 0 : 1;
		}
		if(sum != totalDistances ){
					popConverged[individual] = false;
		}
	}
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
							int * ind,
							int * numWithinBlocks,
							const int * NextPowerTotalDistances,
							const float * __restrict__ bigdistCentroids,
							float * __restrict__ BlockSumWithin
							)
{
	const int gpu_NextPowerTotalDistances = *NextPowerTotalDistances;
	const int gpu_posIndBigDist = *ind * gpu_NextPowerTotalDistances;
	const int gpu_posIndBlockSum = (*ind) * (*numWithinBlocks);
	int idx = blockIdx.x * blockDim.x + threadIdx.x;


	//Specialize BlockReduce for type float
	typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceT;

	//Allocat temporary storage in shared memory
	__shared__ typename BlockReduceT::TempStorage temp_storage_float;

	float result;
	if(idx < gpu_NextPowerTotalDistances)
		result = BlockReduceT(temp_storage_float).Sum(bigdistCentroids[gpu_posIndBigDist + idx]);
	__syncthreads();
	if(threadIdx.x == 0){
		BlockSumWithin[gpu_posIndBlockSum + blockIdx.x] = result;
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
						int * ind,
						int * tamPoblacion,
						float *dataBase,
						float *centroids,
						unsigned char *member_chromosomes,
						float * distCentroids,
						bool * newMapping,
						int * samples_in_k,
						int * newSamples
					)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int d_totalDistances = KMEANS * N_INSTANCES;

	const int tamCentroids = KMEANS * N_FEATURES;
	const int tamDistCentroids = KMEANS * N_INSTANCES;
	const int tamMapping = KMEANS * N_INSTANCES;
	const int tamSamples = KMEANS;
/* -- */
	int posIndMapping = *ind * tamMapping;
	for(int i=threadIdx.x; i < d_totalDistances; i+= blockDim.x){
		newMapping[posIndMapping + i] = false;
	}
	
	int posIndSamples = *ind * tamSamples;
	for(int i=threadIdx.x; i < KMEANS; i+= blockDim.x){
		samples_in_k[posIndSamples + i] = 0;
	}

	int posIndChromosome = *ind * N_FEATURES;
	__syncthreads();
	for (int i = idx; i < N_INSTANCES; i += blockDim.x) {
		float minDist = INFINITY;
		int selectCentroid = -1;
		int pos = N_FEATURES * i;

		for (int k = 0; k < KMEANS; ++k) {	//51
			float sum = 0.0f;
			int posIndCentroids = (*ind * tamCentroids)     + (k * N_FEATURES);
			int posIndDistCentr = (*ind * tamDistCentroids) + (k * N_INSTANCES);
			

//			int posIndCentroids = (2 * tamCentroids)     + (k * N_FEATURES);
//			int posIndDistCentr = (2 * tamDistCentroids) + (k * N_INSTANCES);
//			int posIndChromosome = 2 * N_FEATURES;
			for (int f = 0; f < N_FEATURES; ++f) {
//				if (member_chromosomes[posIndChromosome + f] & 1) {
					//Multiple accesses to global memory. Better if they were in shared memory. OPT
					//sum += (dataBase[pos + f] - centroids[posIndCentroids + f]) * (dataBase[pos + f] - centroids[posIndCentroids + f]);
					sum += 1;
//				}
			}//f

			float euclidean = sqrt(sum);
			distCentroids[posIndDistCentr + i] = euclidean; //Access to global memory. OPT
			if (euclidean < minDist) {
				minDist = euclidean;
				selectCentroid = k;
			}
		}//k
		//newSamples[i]= selectCentroid;
		//samples_in_k[posIndSamples + selectCentroid]++;
		int aux = atomicAdd(samples_in_k + posIndSamples + selectCentroid, 1);

		newMapping[posIndMapping + (selectCentroid * N_INSTANCES) + i] = true;
		__syncthreads();
	}//i
/* -- */
}

__global__ void kernel_2(float *x, int n)
{
	int tid= threadIdx.x + blockIdx.x * blockDim.x;
	for(int i=tid; i<n; i +=blockDim.x * gridDim.x){
		x[i] = sqrt(pow(3.14159,i));
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

	cudaSetDevice(0);
	const int host_tamPoblacion = end - begin;
	const int host_totalDistances = KMEANS * N_INSTANCES;
	const int host_totalCoord = KMEANS * N_FEATURES;
	const int host_nextPowerTotalDistances = nextPowerOfTwo(host_totalDistances);

	const int host_tamCentroids = KMEANS * N_FEATURES;
	const int host_tamDistCentroids = KMEANS * N_INSTANCES;
	const int host_tamMapping = KMEANS * N_INSTANCES;
	const int host_tamSamples = KMEANS;

	//Allocate CPU structures. We use cudaMallocHost to be able to use cuda streams asynchronous memory operations.
	size_t size = host_tamPoblacion * host_totalDistances * sizeof(bool);
	bool *host_mapping;		checkCudaErrors(cudaMallocHost((void **)&host_mapping, size));
	bool *host_newMapping;	checkCudaErrors(cudaMallocHost((void **)&host_newMapping, size));

	size = host_tamPoblacion * KMEANS * N_FEATURES * sizeof(float);
	float * host_centroids;						checkCudaErrors(cudaMallocHost((void **)&host_centroids, size));
	size = host_tamPoblacion * N_FEATURES * sizeof(unsigned char);
	unsigned char * host_member_chromosomes;	checkCudaErrors(cudaMallocHost((void **)&host_member_chromosomes, size));
	
	//No es necesario, borrar
	size = host_tamPoblacion * KMEANS * N_INSTANCES * sizeof(float);
	unsigned char * host_distCentroids;	checkCudaErrors(cudaMallocHost((void **)&host_distCentroids, size));
	
	
	/****************************************Allocate device memory ****************************************/
	//No varía
	float *device_dataBase;
	size_t size_1 = N_INSTANCES * N_FEATURES * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&device_dataBase, size_1));

	//one per stream
	float *device_centroids;	
	size_t size_2 = host_tamPoblacion * KMEANS * N_FEATURES * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&device_centroids, size_2));

	//one per stream
	unsigned char *device_member_chromosomes;	
	size_t size_3 = host_tamPoblacion * N_FEATURES * sizeof(unsigned char);
	checkCudaErrors(cudaMalloc((void **)&device_member_chromosomes, size_3));

	//one per stream
	float * device_distCentroids;	
	size_t size_4 = host_tamPoblacion * KMEANS * N_INSTANCES * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&device_distCentroids, size_4));

	//one per stream
	bool * device_mapping;		
	size_t size_5 = host_tamPoblacion * KMEANS * N_INSTANCES * sizeof(bool);
	checkCudaErrors(cudaMalloc((void **)&device_mapping, size_5));

	//one per stream
	bool * device_newMapping;
	checkCudaErrors(cudaMalloc((void **)&device_newMapping, size_5));		

	//one per stream
	int * device_samples_in_k; 
	size_t size_6 = host_tamPoblacion * KMEANS * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&device_samples_in_k, size_6));

	//all share it
	int * device_NextPowerTotalDistances;
	size_t size_7 = sizeof(int);
	checkCudaErrors(cudaMalloc(&device_NextPowerTotalDistances, size_7));

	//one per stream
	float * device_bigdistCentroids;	
	size_t size_9 = host_tamPoblacion * host_nextPowerTotalDistances * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&device_bigdistCentroids, size_9));

	//one per stream
	int * device_ind;
	size_t size_10 = sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&device_ind, size_10));

	//one per stream
	int * device_numWithinBlocks;
	checkCudaErrors(cudaMalloc((void **)&device_numWithinBlocks, size_10));

	//all share it
	int * device_tamPoblacion;
	size_t size_11 = sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&device_tamPoblacion, size_11));

	//all share it
	bool * device_popConverged;
	size_t size_13 = host_tamPoblacion * sizeof(bool);
	checkCudaErrors(cudaMalloc((void **)&device_popConverged, size_13));

	//all share it
	int * device_newSamples; 
	size_t size_14 = N_INSTANCES * sizeof(int);
	checkCudaErrors(cudaMalloc((void **)&device_newSamples, size_14));


	/****************************************Copy data for all streams ****************************************/

	//database. All streams share it.
	checkCudaErrors(cudaMemcpy(device_dataBase, dataBase, size_1, cudaMemcpyHostToDevice));

	//mapping (one per kernel)
	for (int i = 0; i < host_tamPoblacion * host_totalDistances; ++i) {
		host_mapping[i] = false;
		host_newMapping[i] = false; //borrar
	}
	checkCudaErrors(cudaMemcpy(device_mapping, host_mapping, size_5, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_newMapping, host_newMapping, size_5, cudaMemcpyHostToDevice)); //borrar

	//member_chromosomes
	for(int ind=begin; ind < end; ++ind){
		for(int i=0; i < N_FEATURES; i++){
			int pos = (ind*host_tamPoblacion)+i;
			host_member_chromosomes[pos] = pop[ind].chromosome[i];
		}
	}

/* -CHECK- * /
	//NO HACE FALTA, BORRAR:
	for (int ind = begin; ind < end; ++ind) {
		for(int i=0; i<N_FEATURES; i++){
			printf("\ngpu member_chromosomes[%d]= %u", i, host_member_chromosomes[i]);
		}
		printf("\n----");
	}
/* -CHECK- */

	checkCudaErrors(cudaMemcpy(device_member_chromosomes, host_member_chromosomes, size_3, cudaMemcpyHostToDevice));

/* -CHECK- * /
	for(int ind=begin; ind < end; ++ind){
		for(int i=0; i < N_FEATURES; i++){
			int pos = (ind*host_tamPoblacion)+i;
			printf("\nmember_chromosomes[%d][%d]= %u, original: %u", ind, i, host_member_chromosomes[pos], pop[ind].chromosome[i]);
		}
		
	}
/* -CHECK- */

	//centroids (one per kernel)
	for(int ind=begin; ind < end; ++ind){
		for (int k = 0; k < KMEANS; ++k) {
			int posDataBase = selInstances[k] * N_FEATURES;
			int posCentr = k * N_FEATURES + (ind * host_totalCoord);

			for (int f = 0; f < N_FEATURES; ++f) {
//				printf("\nEscribiendo %f en la posición %d de host_centroids\n", dataBase[posDataBase + f], posCentr + f);
//				if (pop[ind].chromosome[f] & 1) {
					host_centroids[posCentr + f] = dataBase[posDataBase + f];
				}
//			}
		}
//		printf("\n-----------");
	}
	checkCudaErrors(cudaMemcpy(device_centroids, host_centroids, size_2, cudaMemcpyHostToDevice));

	//distCentroids (no es necesario, borrar)
	for(int i=0; i<host_tamPoblacion * host_totalDistances; i++){
		host_distCentroids[i]=0;
	}
	size = host_tamPoblacion * KMEANS * N_INSTANCES * sizeof(float);
	checkCudaErrors(cudaMemcpy(device_distCentroids, host_distCentroids, size, cudaMemcpyHostToDevice));


	//tamPoblacion
	checkCudaErrors(cudaMemcpy(device_tamPoblacion, &host_tamPoblacion, size_11, cudaMemcpyHostToDevice));


	bool * host_popConverged = (bool*) malloc(host_tamPoblacion * sizeof(bool));
	for(int i=0; i<host_tamPoblacion; i++){
		host_popConverged[i]=false;
	}
	checkCudaErrors(cudaMemcpy(device_popConverged, host_popConverged, size_13, cudaMemcpyHostToDevice));

    /* PRUEBAS CON LOS STREAMS * /

    const int num_streams = 16; //Theoric limit in Fermi architecture (compute capability 2.0+)
    cudaStream_t streams[num_streams];

    float * data[num_streams];

    for(int i=0; i < num_streams; i++){
        cudaStreamCreate(&streams[i]);
        
        cudaMalloc(&data[i], N*sizeof(float));

        //launch one worker kernel per stream
        kernel<<< 1, 64, 0, streams[i]>>>(data[i], N);

        //launch a dummy kernel on the default stream
//        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    /* PRUEBAS CON LOS STREAMS */


	//Decide number of blocks and threads for each parallel section.
	cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

	unsigned int numEuclideanThreadsPerBlock = BLOCK_SIZE;
	unsigned int numEuclideanBlocks = ((N_INSTANCES+numEuclideanThreadsPerBlock)-1) / numEuclideanThreadsPerBlock;

	unsigned int numUpdateThreadsPerBlock = BLOCK_SIZE_2;
	unsigned int numUpdateBlocks = ((N_FEATURES+numUpdateThreadsPerBlock)-1) / numUpdateThreadsPerBlock;
	//unsigned int numEuclideanBlocks = 1;
	if(numEuclideanBlocks==0){numEuclideanBlocks=1;}
	if(numEuclideanBlocks==0){numEuclideanBlocks=1;}

	//In order for CUBLAS to support reduction, numWithinThreadsPerBlock must be a power of two.
	unsigned int numWithinThreadsPerBlock = BLOCK_SIZE;
	unsigned int numWithinBlocks = ((host_totalDistances+host_totalDistances)-1) / BLOCK_SIZE;

	unsigned int numSwapThreadsPerBlock = BLOCK_SIZE;
	unsigned int numSwapBlocks = ((host_totalDistances+host_totalDistances)-1) / BLOCK_SIZE;

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
/* -- */
	bool gpu_converged = false;
//	float gpu_distCentroids[host_tamPoblacion * KMEANS * N_INSTANCES];
//	bool gpu_mapping[host_tamPoblacion * KMEANS * N_INSTANCES];
//	bool gpu_newMapping[host_tamPoblacion * KMEANS * N_INSTANCES];
//	int gpu_samples_in_k[host_tamPoblacion * KMEANS];
//	float gpu_centroids[host_tamPoblacion * KMEANS * N_FEATURES];

	size = host_tamPoblacion * host_tamCentroids * sizeof(float);
	float * gpu_centroids; checkCudaErrors(cudaMallocHost((void **)&gpu_centroids, size));

	size = host_tamPoblacion * host_tamCentroids * sizeof(float);
	float * results_centroids; checkCudaErrors(cudaMallocHost((void **)&results_centroids, size));

	size = host_tamPoblacion * host_tamDistCentroids * sizeof(float);
	float * results_distCentroids; checkCudaErrors(cudaMallocHost((void **)&results_distCentroids, size));
	
	size = host_tamPoblacion * host_tamSamples * sizeof(int);
	int * results_samples; checkCudaErrors(cudaMallocHost((void **)&results_samples, size));

	size = host_tamPoblacion * host_tamMapping * sizeof(bool);
	bool * results_mapping; checkCudaErrors(cudaMallocHost((void **)&results_mapping, size));

	size = host_tamPoblacion * host_tamMapping * sizeof(bool);
	bool * results_newMapping; checkCudaErrors(cudaMallocHost((void **)&results_newMapping, size));

	//Allocate extra device necessary structures 
	//one per stream
	float * device_BlockSumWithin;
	size_t size_12 = host_tamPoblacion * numWithinBlocks * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&device_BlockSumWithin, size_12));

	//In order for CUBLAS to support reduction, size of the structure used must match the number of threads we are using.
	//numWithinThreadsPerBlock is a power of two.
	size= host_tamPoblacion * host_nextPowerTotalDistances * sizeof(float);
	float * gpu_bigdistCentroids;	checkCudaErrors(cudaMallocHost((void **)&gpu_bigdistCentroids, size));

	float * gpu_BlockSumWithin;	checkCudaErrors(cudaMallocHost((void **)&gpu_BlockSumWithin, size_12));

	size = host_tamPoblacion * host_tamDistCentroids * sizeof(float);
	float * gpu_distCentroids;	checkCudaErrors(cudaMallocHost((void **)&gpu_distCentroids, size));

	size = host_tamPoblacion * host_tamMapping * sizeof(bool);
	bool * gpu_mapping;	checkCudaErrors(cudaMallocHost((void **)&gpu_mapping, size));

	size = host_tamPoblacion * host_tamMapping * sizeof(bool);
	bool * gpu_newMapping;	checkCudaErrors(cudaMallocHost((void **)&gpu_newMapping, size));
	
	//------------------------------

	size = host_tamCentroids * sizeof(float);
	float * gpu_Indcentroids;	checkCudaErrors(cudaMallocHost((void **)&gpu_Indcentroids, size));

	size = host_tamSamples * sizeof(int);
	int * gpu_Indsamples_in_k;	checkCudaErrors(cudaMallocHost((void **)&gpu_Indsamples_in_k, size));

	size = host_tamPoblacion * sizeof(float);
	float * gpu_SumWithin_2;	checkCudaErrors(cudaMallocHost((void **)&gpu_SumWithin_2, size));

	float * gpu_SumInter;	checkCudaErrors(cudaMallocHost((void **)&gpu_SumInter, size));

	bool allConverged = false;
	int nVueltas=0;

	const int num_streams = host_tamPoblacion;
	cudaStream_t streams[num_streams];

//	float *data[num_streams];
//	for(int i=0; i< num_streams; i++){
//		cudaMalloc(&data[i], N*sizeof(float));
//	}
	while(!allConverged){
		nVueltas++;
//		printf("\nnVueltas=%d", nVueltas);
		// Evaluate all the individuals
//		for (int ind = begin; ind < end; ++ind) {
			for(int i=0; i< num_streams; i++){				//52 GPU
//				totalIndProcessed++;
//				cudaStreamCreate(&streams[i]);
				//Aquí se corren todos los streams una vez. HACER
//				kernel_2<<<1, 64, 0, streams[i]>>>(data[i], N);
//				kernel_2<<<1, 1>>>(0, 0);
/* -- */
				//checkCudaErrors(cudaMemcpy(device_ind, &ind, sizeof(int), cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMemcpy(device_ind, &i, sizeof(int), cudaMemcpyHostToDevice));
				cudaStreamCreate(&streams[i]);
				//Use GPU for the heavy computing part.
//				printf("\nLanzando %d bloques con %d hebras cada uno en el stream %d", numEuclideanBlocks, numEuclideanThreadsPerBlock, i);
/* -- */
				//Funciona bien, sin problemas de concurrencia ni dimensionalidad
				//RECORDAR AJUSTAR EL BLOCKSIZE
					cuda_Convergence_Euclidean <<< numEuclideanBlocks, numEuclideanThreadsPerBlock, 0, streams[i] >>> (
																		device_ind,
																		device_tamPoblacion,
																		device_dataBase, 
																		device_centroids, 
																		device_member_chromosomes, 
																		device_distCentroids, 
																		device_newMapping,
																		device_samples_in_k,
																		device_newSamples
																		);
			}
/* -- * /			
				for(int i=0; i< num_streams; i++){
					cudaStreamCreate(&streams[i]);
					//JUST 1 THREAD FUNCIONA BIEN
					cudaDeviceSynchronize();
					cuda_Convergence_Check <<<1, 1, 0, streams[i]>>>( //50
																		device_ind,
																		device_mapping,
																		device_newMapping,
																		device_popConverged
																		);
				}																		
/* -- */
//				cudaDeviceSynchronize();
			for(int i=0; i< num_streams; i++){
				cudaStreamCreate(&streams[i]);
				cuda_Convergence_Update <<< numUpdateBlocks, numUpdateThreadsPerBlock, 0, streams[i]>>>(
																		device_ind,
																		device_popConverged,
																		device_centroids,
																		device_member_chromosomes,
																		device_samples_in_k,
																		device_newMapping,
																		device_dataBase
																		);
			}
/* -- */
//			cudaDeviceSynchronize();
			for(int i=0; i< num_streams; i++){
				cudaStreamCreate(&streams[i]);
				cuda_Convergence_SwapMappings <<< numSwapBlocks, numSwapThreadsPerBlock, 0, streams[i] >>>(
																		device_ind,
																		device_popConverged,
																		device_mapping,
																		device_newMapping
																		);
			}
//			cudaDeviceSynchronize();
			
/* -- */
//			}//ind Euclidean, Converged and Update
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(host_popConverged, 	 device_popConverged, 		size_13, cudaMemcpyDeviceToHost));	

		/* -- * /
		for(int i=0; i<host_tamPoblacion; i++){
			printf("\n gpu popConverged [%d]=%d", i, host_popConverged[i]);
		}
		printf("\n--------");
		/* -- */

		//Anyone left to converge?
/* -- * /
		allConverged = true;
		for(int i=0; i<host_tamPoblacion && allConverged; i++){
			if(!host_popConverged[i]){
				allConverged = false;
			}
		}
		

		if(allConverged){
			printf("\nHa convergío en la vuelta %d", nVueltas);
		}
/* -- */



		
/* -- * /
		checkCudaErrors(cudaMemcpy(gpu_distCentroids,device_distCentroids, 	size_4, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(gpu_newMapping, 	 device_newMapping,		size_5, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(gpu_mapping, 	 device_mapping, 		size_5, cudaMemcpyDeviceToHost));

/* -- * /
		for (int ind = begin; ind < end; ++ind) { 
			int posIndCentroids =  ind * host_tamCentroids;
			int posIndDistCentr =  ind * host_tamDistCentroids;
			int posIndChromosome = ind * N_FEATURES;
			int posIndMapping = ind * host_tamMapping;
			int posIndSamples = ind * host_tamSamples;
/* -- * /

			//Has the algorithm converged? We use CPU.
			popConverged[ind] = true;	//53 GPU
			int sum=0;
			for(int i=0; i<host_totalDistances; i++){
				sum+= (gpu_mapping[posIndMapping + i] != gpu_newMapping[posIndMapping + i]) ? 0 : 1;
			}
			if(sum != host_totalDistances ){
						popConverged[ind] = false;
			}

			/* -- * /
			for(int i=0; i<host_tamPoblacion; i++){
				printf("\n gpu popConverged [%d]=%d", i, popConverged[i]);
			}
			printf("\n--------");
			/* -- */

/* -- * /	//49
			if (!popConverged[ind]) {   
				nVueltas++;
				size = host_tamCentroids * sizeof(float);
				checkCudaErrors(cudaMemcpy(gpu_Indcentroids, 	device_centroids, 						size, cudaMemcpyDeviceToHost));
				
				size = host_tamSamples * sizeof(int);
				checkCudaErrors(cudaMemcpy(gpu_Indsamples_in_k, device_samples_in_k + posIndSamples, 	size, cudaMemcpyDeviceToHost));

				// Update the position of the centroids
				for (int k = 0; k < KMEANS; ++k) {
					int posCentroids = k * N_FEATURES;
					int posMap = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						float sum = 0.0f;
						if (pop[ind].chromosome[f] & 1) {
							for (int i = 0; i < N_INSTANCES; ++i) {
								if (gpu_newMapping[posIndMapping + posMap + i]) {
									sum += dataBase[(N_FEATURES * i) + f];
								}
							}
							if(gpu_Indsamples_in_k[posIndSamples + k] == 0){
								gpu_Indcentroids[posIndCentroids + posCentroids + f] = 0;
							}else{
								gpu_Indcentroids[posIndCentroids + posCentroids + f] = sum / gpu_Indsamples_in_k[posIndSamples + k];
							}
						}//if chromosome
					}//for nfeatures
				}//for KMEANS
/* -- * /
				//New centroids, thanks to CPU work
				size = host_tamCentroids * sizeof(float);
				checkCudaErrors(cudaMemcpy(device_centroids + posIndCentroids, 	gpu_Indcentroids, size, 	cudaMemcpyHostToDevice));
/* -- * /
			}//!converged
/* -- * /
			//Anyone left to converge?
			allConverged = true;
			for(int i=0; i<host_tamPoblacion && allConverged; i++){
				if(!popConverged[i]){
					allConverged = false;
				}
			}
/* -- * /
		}//ind Converged
		// Swap GPU mapping tables 
		checkCudaErrors(cudaMemcpy(device_mapping, 		gpu_newMapping, size_5, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(device_newMapping, 	gpu_mapping, size_5, 	cudaMemcpyHostToDevice));
/* -- */
//		allConverged=true;
		if(nVueltas == 50){
			allConverged=true;
		}
		
/* -- */
	}//while ninguno sin converger		//52
	
//	cudaDeviceSynchronize();
//	for(int i=0; i< num_streams; i++){
//		cudaStreamDestroy(streams[i]);
//	}
	
/* -- */
	checkCudaErrors(cudaMemcpy(gpu_distCentroids,device_distCentroids, 	size_4, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(gpu_mapping, 	 device_mapping, 		size_5, cudaMemcpyDeviceToHost));

		/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) *************/
	for (int ind = begin; ind < end; ++ind) {
		int posIndCentroids =  ind * host_tamCentroids;
		int posIndDistCentr =  ind * host_tamDistCentroids;
		int posIndChromosome = ind * N_FEATURES;
		int posIndMapping = ind * host_tamMapping;
		int posIndSamples = ind * host_tamSamples;

		int posIndBigDist = ind * host_nextPowerTotalDistances;
		int k=0;
		for (; k < host_totalDistances; ++k) { 	//50  GPU
//				printf("\nSe va a sumar %f", gpu_distCentroids[posIndDistCentr + k] * gpu_mapping[posIndMapping + k]); 
				gpu_bigdistCentroids[posIndBigDist + k] = gpu_distCentroids[posIndDistCentr + k] * gpu_mapping[posIndMapping + k];			
//				gpu_bigdistCentroids[posIndBigDist + k] = 1.0f;
		}
		for(; k < host_nextPowerTotalDistances; k++){
//			gpu_bigdistCentroids[k] = 0.0f;
			gpu_bigdistCentroids[posIndBigDist + k] = 0.0f;
		}

	}//ind rellenar bigdist
/* -CHECK-* /
//		printf("\nSe va a mandar a la GPU:");
		for(int k=0; k < host_tamPoblacion * host_nextPowerTotalDistances; k++){
//			printf("\n gpu_bigdistCentroids[%d]=%f", k, gpu_bigdistCentroids[k]);
		}
/* -CHECK-*/

	checkCudaErrors(cudaMemcpy(device_bigdistCentroids,	gpu_bigdistCentroids, 	size_9, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(device_NextPowerTotalDistances, &host_nextPowerTotalDistances, sizeof(int), cudaMemcpyHostToDevice));	
	for (int ind = begin; ind < end; ++ind) {
		checkCudaErrors(cudaMemcpy(device_ind, &ind, sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(device_numWithinBlocks, &numWithinBlocks, sizeof(int), cudaMemcpyHostToDevice));
		// Within-cluster
		cuda_WithinCluster <<< numWithinBlocks, numWithinThreadsPerBlock>>> (
														device_ind,
														device_numWithinBlocks,
														device_NextPowerTotalDistances, 
														device_bigdistCentroids,
														device_BlockSumWithin
														);
		cudaDeviceSynchronize();
	}//ind

	checkCudaErrors(cudaMemcpy(gpu_BlockSumWithin, 	device_BlockSumWithin, 	size_12, cudaMemcpyDeviceToHost));
	for (int ind = begin; ind < end; ++ind) {
		int posIndBlockSum = ind * numWithinBlocks;
		
		for(int i=0; i < numWithinBlocks; i++){
//			printf("\n  sumando %f", gpu_BlockSumWithin[posIndBlockSum + i]);
			gpu_SumWithin_2[ind] += gpu_BlockSumWithin[posIndBlockSum + i];
		}
	}//ind
	
	// Inter-cluster  //TODO: paralelize this if it's worth the effort
	checkCudaErrors(cudaMemcpy(gpu_centroids, 		device_centroids,		size_2,  cudaMemcpyDeviceToHost));
	for (int ind = begin; ind < end; ++ind) {
		int posIndCentroids =  ind * host_tamCentroids;
		
		for (int k = 0; k < KMEANS; ++k) {
			int gpu_posCentroids = k * N_FEATURES;
			for (int i = gpu_posCentroids + N_FEATURES; i < host_totalCoord; i += N_FEATURES) {
				float sum = 0.0f;
				for (int f = 0; f < N_FEATURES; ++f) {
					if (pop[ind].chromosome[f] & 1) {
						sum += (gpu_centroids[posIndCentroids + gpu_posCentroids + f] 
							-   gpu_centroids[posIndCentroids + i + f]) 
							*  (gpu_centroids[posIndCentroids + gpu_posCentroids + f] 
							-   gpu_centroids[posIndCentroids + i + f]);
					}
				}
				gpu_SumInter[ind] += sqrt(sum);
			}
		}
	}//ind
	for (int ind = begin; ind < end; ++ind) {
		// First objective function (Within-cluster sum of squares (WCSS))
		pop[ind].fitness[0] = gpu_SumWithin_2[ind];

		// Second objective function (Inter-cluster sum of squares (ICSS))
		pop[ind].fitness[1] = gpu_SumInter[ind];


//		printf("\ngpu sumWithin[%d]=%f", ind, gpu_SumWithin_2[ind]);
//		printf("\ngpu sumInter[%d]=%f", ind, gpu_SumInter[ind]);

		// Third objective function (Number of selected features)
		//pop[ind].fitness[2] = (float) nSelFeatures;

//		checkCudaErrors(cudaFree(d_BlockSumWithin));
		//WCSS and ICSS minimization process
	}//ind
/* -- */
	checkCudaErrors(cudaMemcpy(results_centroids,		device_centroids, 		size_2, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(results_distCentroids,	device_distCentroids, 	size_4, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(results_samples,			device_samples_in_k, 	size_6, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(results_mapping,			device_mapping, 		size_5, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(results_newMapping,		device_newMapping, 		size_5, cudaMemcpyDeviceToHost));

/* -CHECK- * /			//CHECKS
	float auxSumaCentroids=0;
	float auxSumaDist=0;
	int auxSumaSamples=0;
	int auxSumaMapping=0;
	int auxSumaNewMapping=0;
	int auxCentroidsBlancos=0;
	for(int i=0; i<host_tamPoblacion * KMEANS*N_FEATURES; i++){
		auxSumaCentroids += results_centroids[i];
		if(results_centroids[i]==0){
			auxCentroidsBlancos++;
		}
//		printf("\nGPU results_centroids[%d]= %f", i, results_centroids[i]);
	}
	for(int i=0; i<host_tamPoblacion * KMEANS*N_INSTANCES; i++){
		auxSumaDist += results_distCentroids[i];
//			printf("\nGPU results_distCentroids[%d]= %f", i, results_distCentroids[i]);
	}	
	for(int i=0; i<host_tamPoblacion * host_tamSamples; i++){
		auxSumaSamples += results_samples[i];
//		printf("\nGPU samples_in_k[%d]=%d", i, results_samples[i]);
	}
	for(int i=0; i<host_tamPoblacion * host_tamMapping; i++){
		auxSumaMapping += results_mapping[i];
//		printf("\nGPU mapping[%d]=%d", i, results_mapping[i]);
	}
	for(int i=0; i<host_tamPoblacion * host_tamMapping; i++){
		auxSumaNewMapping += results_newMapping[i];
//		printf("\nGPU newMapping[%d]=%d", i, results_newMapping[i]);
	}

	printf("\nnVueltas=%d", nVueltas);

	printf("\nGPU: Suma total de centroids: %f", auxSumaCentroids);
	printf("\nTEST: Posiciones en blanco de centroids: %d", auxCentroidsBlancos);
	printf("\nGPU: Suma total de distCentroids: %f", auxSumaDist);
	printf("\nGPU: Suma total de samples: %d", auxSumaSamples);
	printf("\nGPU: Suma total de mapping: %d", auxSumaMapping);
	printf("\nGPU: Suma total de newMapping: %d", auxSumaNewMapping);
	printf("\nvalor de host_nextPowerTotalDistances= %d", host_nextPowerTotalDistances);
/* -CHECK- */

/* -- */
	// Resources used are released
	checkCudaErrors(cudaFree(device_dataBase));
	checkCudaErrors(cudaFree(device_centroids));
	checkCudaErrors(cudaFree(device_member_chromosomes));
	checkCudaErrors(cudaFree(device_distCentroids));
	checkCudaErrors(cudaFree(device_mapping));
	checkCudaErrors(cudaFree(device_newMapping));
	checkCudaErrors(cudaFree(device_samples_in_k));
	checkCudaErrors(cudaFree(device_NextPowerTotalDistances));
	checkCudaErrors(cudaFree(device_bigdistCentroids));

	checkCudaErrors(cudaFree(device_ind));
	checkCudaErrors(cudaFree(device_numWithinBlocks));
	checkCudaErrors(cudaFree(device_tamPoblacion));
	checkCudaErrors(cudaFree(device_popConverged));
	checkCudaErrors(cudaFree(device_newSamples));
/* -- * /
/* -- */

}//CUDA_kmeans

/**
 * @brief Sequential K-means algorithm which minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS).
 * @param pop Current population.
 * @param begin The first individual to evaluate.
 * @param end The "end-1" position is the last individual to evaluate.
 * @param selInstances The instances choosen as initial centroids.
 * @param dataBase The database which will contain the instances and the features.
 */

void test_cpu_kmeans(individual *pop, const int begin, const int end, const int *const selInstances, const float *const dataBase) {
	const int host_tamPoblacion = end - begin;
	const int host_totalDistances = KMEANS * N_INSTANCES;
	const int host_totalCoord = KMEANS * N_FEATURES;
	const int host_nextPowerTotalDistances = nextPowerOfTwo(host_totalDistances);

	const int host_tamCentroids = KMEANS * N_FEATURES;
	const int host_tamDistCentroids = KMEANS * N_INSTANCES;
	const int host_tamMapping = KMEANS * N_INSTANCES;
	const int host_tamSamples = KMEANS;
	const int totalCoord = KMEANS * N_FEATURES;

	bool *mapping = (bool*) malloc(host_tamPoblacion * host_tamMapping * sizeof(bool));
	bool *newMapping = (bool*) malloc(host_tamPoblacion * host_tamMapping * sizeof(bool));

	float centroids[host_tamPoblacion * host_tamCentroids];

	// The centroids will have the selected features of each individual
	for(int ind=begin; ind < end; ++ind){
		for (int k = 0; k < KMEANS; ++k) {
			int posDataBase = selInstances[k] * N_FEATURES;
			int posCentr = k * N_FEATURES + (ind * host_totalCoord);

			for (int f = 0; f < N_FEATURES; ++f) {
//				printf("\nEscribiendo %f en la posición %d de host_centroids\n", dataBase[posDataBase + f], posCentr + f);
//				if (pop[ind].chromosome[f] & 1) {
					centroids[posCentr + f] = dataBase[posDataBase + f];
				}
//			}
		}
//		printf("\n-----------");
	}

	// Initialize the array of minimum distances and the mapping table
	const int totalDist = KMEANS * N_INSTANCES;
	float distCentroids[host_tamPoblacion * KMEANS * N_INSTANCES];
	int samples_in_k[host_tamPoblacion * KMEANS];

	// Initialize the mapping table
	for (int i = 0; i < host_tamPoblacion * host_totalDistances; ++i) {
		mapping[i] = false;
	}
	for(int i=0; i<host_tamPoblacion * host_totalDistances; i++){
		distCentroids[i]=0;
	}

/* -CHECK- * /
	//NO HACE FALTA, BORRAR:
	for (int ind = begin; ind < end; ++ind) {
		for(int i=0; i<N_FEATURES; i++){
//			printf("\ntest member_chromosomes[%d]= %u", i, pop[ind].chromosome[i]);
		}
		printf("\n----");
	}
/* -CHECK- */

	// Evaluate all the individuals
//	for (int ind = begin; ind < end; ++ind) {
	bool * popConverged = (bool*) malloc(host_tamPoblacion * sizeof(bool));
	for(int i=0; i<host_tamPoblacion; i++){
		popConverged[i]=false;
	}
	bool allConverged = false;

	int nVueltas=0;
	while(!allConverged){
		nVueltas++;
		printf("\nnVueltas=%d", nVueltas);

		/******************** Convergence process *********************/
		
//		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !converged; ++maxIter) {	//52 test
		for (int ind = begin; ind < end; ++ind) {
			
			// The mapping table is cleaned in each iteration
			int posIndMapping = ind * host_tamMapping;
			for (int i = 0; i < totalDist; ++i) {
				newMapping[posIndMapping + i] = false;
			}
			
			int posIndSamples = ind * host_tamSamples;
			for (int i = 0; i < KMEANS; ++i) {
				samples_in_k[posIndSamples + i] = 0;
			}

			// Calculate all distances (Euclidean distance) between each instance and the centroids
			for (int i = 0; i < N_INSTANCES; ++i) {
				float minDist = INFINITY;
				int selectCentroid = -1;
				int pos = N_FEATURES * i;

				for (int k = 0; k < KMEANS; ++k) {
					float sum = 0.0f;
					int posIndCentroids = (ind * host_tamCentroids)     + (k * N_FEATURES);
					int posIndDistCentr = (ind * host_tamDistCentroids) + (k * N_INSTANCES);
					int posIndChromosome = ind * N_FEATURES;
					for (int f = 0; f < N_FEATURES; ++f) {
//						if (pop[ind].chromosome[f] & 1) {
//							sum += (dataBase[pos + f] - centroids[posIndCentroids + f]) * (dataBase[pos + f] - centroids[posIndCentroids + f]);
							sum += 1;
//						}
					}

					float euclidean = sqrt(sum);
					distCentroids[posIndDistCentr + i] = euclidean;
					if (euclidean < minDist) {
						minDist = euclidean;
						selectCentroid = k;
					}
				}

				newMapping[posIndMapping + (selectCentroid * N_INSTANCES) + i] = true;
//				printf("\nActualizando samples_in_k[%d]", posIndSamples + selectCentroid);
				samples_in_k[posIndSamples + selectCentroid]++;
			}
		}//ind Euclidean

/* -- */
		for (int ind = begin; ind < end; ++ind) {
			int posIndCentroids =  ind * host_tamCentroids;
			int posIndDistCentr =  ind * host_tamDistCentroids;
			int posIndChromosome = ind * N_FEATURES;
			int posIndMapping = ind * host_tamMapping;
			int posIndSamples = ind * host_tamSamples;
/* -- * /
			//Has the algorithm converged? We use CPU.
			popConverged[ind] = true;			//53 test
			int sum=0;		
			for(int i=0; i<host_totalDistances; i++){
				sum+= (mapping[posIndMapping + i] != newMapping[posIndMapping + i]) ? 0 : 1;
			}
			if(sum != host_totalDistances ){
						popConverged[ind] = false;
			}
			/* -- * /
			for(int i=0; i<host_tamPoblacion; i++){
				printf("\n test popConverged [%d]=%d", i, popConverged[i]);
			}
			printf("\n--------");		
			/* -- */
/* -- */
			if (!popConverged[ind]) {
				// Update the position of the centroids
				for (int k = 0; k < KMEANS; ++k) {
					int posCentr = k * N_FEATURES;
					int posMap = k * N_INSTANCES;
					for (int f = 0; f < N_FEATURES; ++f) {
						float sum = 1.0f;
//						if (pop[ind].chromosome[f] & 1) {
							for (int i = 0; i < N_INSTANCES; ++i) {
								if (newMapping[posIndMapping + posMap + i]) {
									sum += dataBase[(N_FEATURES * i) + f];
//									sum += 1.0f;
								}
							}

							if(samples_in_k[posIndSamples + k] == 0){
								centroids[(ind * host_tamCentroids) + posCentr + f] = 0;
							}else{
								centroids[(ind * host_tamCentroids) + posCentr + f] = sum / samples_in_k[posIndSamples + k];
							}
//						}
					}
				}
			}
/* -- * /
			//Anyone left to converge?
			allConverged = true;
			for(int i=0; i<host_tamPoblacion && allConverged; i++){
				if(!popConverged[i]){
					allConverged = false;
				}
			}
			if(allConverged){
				printf("\nHa convergío en la vuelta %d", nVueltas);
			}
/* -- */			
		}//ind Converged
/* -- */
		// Swap mapping tables
		bool *aux = newMapping;
		newMapping = mapping;
		mapping = aux;
/* -- */
		
		if(nVueltas==50){
			allConverged=true;
		}
	}//ninguno sin converger		//52

		/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) *************/

	float sumWithin[host_tamPoblacion];
	float sumInter[host_tamPoblacion];
	
	for (int ind = begin; ind < end; ++ind) {
			int posIndCentroids =  ind * host_tamCentroids;
			int posIndDistCentr =  ind * host_tamDistCentroids;
			int posIndChromosome = ind * N_FEATURES;
			int posIndMapping = ind * host_tamMapping;
			int posIndSamples = ind * host_tamSamples;

		for (int k=0; k < host_totalDistances; ++k) { 	//50  test
//				printf("\nSe va a sumar %f", gpu_distCentroids[posIndDistCentr + k] * gpu_mapping[posIndMapping + k]); 
				sumWithin[ind] += distCentroids[posIndDistCentr + k] * mapping[posIndMapping + k];			
//				sumWithin[ind] += distCentroids[posIndDistCentr + k] = 1.0f;
		}
	}//ind
	for (int ind = begin; ind < end; ++ind) {
		int posIndCentroids =  ind * host_tamCentroids;
		for (int k = 0; k < KMEANS; ++k) {
			int posCentr = (k * N_FEATURES);
			// Inter-cluster
			for (int i = posCentr + N_FEATURES; i < totalCoord; i += N_FEATURES) {
				float sum = 0.0f;
				for (int f = 0; f < N_FEATURES; ++f) {
					if (pop[ind].chromosome[f] & 1) {
						sum += (centroids[posIndCentroids + posCentr + f]
						    -  centroids[i + f])
						    * (centroids[posIndCentroids +  posCentr + f]
						    -  centroids[i + f]);
					}
				}
				sumInter[ind] += sqrt(sum);
			}
		}
	}//for each individual

	for (int ind = begin; ind < end; ++ind) {

		// First objective function (Within-cluster sum of squares (WCSS))
		pop[ind].fitness[0] = sumWithin[ind];

		// Second objective function (Inter-cluster sum of squares (ICSS))
		pop[ind].fitness[1] = sumInter[ind];

		printf("\ntest sumWithin[%d]=%f", ind, sumWithin[ind]);
		printf("\ntest sumInter[%d]=%f", ind, sumInter[ind]);

		// Third objective function (Number of selected features)
		//pop[ind].fitness[2] = (float) nSelFeatures;
	}//ind

	
/* -- * /
	float auxSumaCentroids=0;			//CHECKS
	float auxSumaDist=0;
	int auxSumaSamples=0;
	int auxSumaMapping=0;
	int auxSumaNewMapping=0;
	int auxCentroidsBlancos=0;
		
	for(int i=0; i<host_tamPoblacion * host_tamCentroids; i++){
		auxSumaCentroids += centroids[i];
		if(centroids[i]==0){
			auxCentroidsBlancos++;
		}
//		printf("\ntest centroids[%d]= %f", i, centroids[i]);
	}	
	//Euclidean
	for(int i=0; i<host_tamPoblacion * host_tamDistCentroids; i++){
		auxSumaDist += distCentroids[i];
//		printf("\ntest test_distCentroids[%d]= %f", i, distCentroids[i]);
	}
	for(int i=0; i<host_tamPoblacion * KMEANS; i++){
		auxSumaSamples += samples_in_k[i];
//		printf("\ntest samples_in_k[%d]=%d", i, samples_in_k[i]);
	}
	for(int i=0; i<host_tamPoblacion * host_tamMapping; i++){
		auxSumaMapping += mapping[i];
//		printf("\ntest mapping[%d]=%d", i, mapping[i]);
	}
	for(int i=0; i<host_tamPoblacion * host_tamMapping; i++){
		auxSumaNewMapping += newMapping[i];
//		printf("\ntest newMapping[%d]=%d", i, newMapping[i]);
	}

	printf("\nnVueltas=%d", nVueltas);
	//(WCSS and ICSS)
	
		

	printf("\nTEST: Suma total de centroids: %f", auxSumaCentroids);
	printf("\nTEST: Posiciones en blanco de centroids: %d", auxCentroidsBlancos);
	printf("\nTEST: Suma total de distCentroids: %f", auxSumaDist);
	printf("\nTEST: Suma total de samples: %d", auxSumaSamples);
	printf("\nTEST: Suma total de mapping: %d", auxSumaMapping);
	printf("\nTEST: Suma total de newMapping: %d", auxSumaNewMapping);
	printf("\nvalor de host_nextPowerTotalDistances= %d", host_nextPowerTotalDistances);


/* -- */
	// Resources used are released
	free(mapping);
	free(newMapping);
/* -- */
}

/**
 * @brief Sequential K-means algorithm which minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS).
 * @param pop Current population.
 * @param begin The first individual to evaluate.
 * @param end The "end-1" position is the last individual to evaluate.
 * @param selInstances The instances choosen as initial centroids.
 * @param dataBase The database which will contain the instances and the features.
 */
void cpu_kmeans(individual *pop, const int begin, const int end, const int *const selInstances, const float *const dataBase) {
	const int host_tamPoblacion = end - begin;
	const int host_totalDistances = KMEANS * N_INSTANCES;
	const int host_totalCoord = KMEANS * N_FEATURES;
	const int host_nextPowerTotalDistances = nextPowerOfTwo(host_totalDistances);

	const int host_tamCentroids = KMEANS * N_FEATURES;
	const int host_tamDistCentroids = KMEANS * N_INSTANCES;
	const int host_tamMapping = KMEANS * N_INSTANCES;
	const int host_tamSamples = KMEANS;
	const int totalCoord = KMEANS * N_FEATURES;

	bool *mapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));
	bool *newMapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));
	const int tamPoblacion = end - begin;

	// Evaluate all the individuals
	int nVueltas=0;
	for (int ind = begin; ind < end; ++ind) {
		const int totalCoord = KMEANS * N_FEATURES;
		float centroids[KMEANS * N_FEATURES];

		// The centroids will have the selected features of the individual
		for (int k = 0; k < KMEANS; ++k) {
			int posDataBase = selInstances[k] * N_FEATURES;
			int posCentr = k * N_FEATURES;

			for (int f = 0; f < N_FEATURES; ++f) {
//				if (pop[ind].chromosome[f] & 1) {
					centroids[posCentr + f] = dataBase[posDataBase + f];
				}
//			}
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
		int contadorr=0;
		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !converged; ++maxIter) {	//52 cpu

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
							//sum +=1;
						}
					}
					

					float euclidean = sqrt(sum);
					distCentroids[posDistCentr + i] = euclidean;
					if (euclidean < minDist) {
						contadorr++;
						minDist = euclidean;
						selectCentroid = k;
					}
				}

				newMapping[(selectCentroid * N_INSTANCES) + i] = true;
				samples_in_k[selectCentroid]++;
			}
/* -- * /	
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
/* -- */

//			if (!converged) {
				nVueltas++;
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

							if(samples_in_k[k] == 0){
								centroids[posCentr + f] = 0;
							}else{
								centroids[posCentr + f] = sum / samples_in_k[k];
							}
						}
					}
//				}

				// Swap mapping tables
				bool *aux = newMapping;
				newMapping = mapping;
				mapping = aux;
			}
/* -- */
			if(nVueltas == 50){
				converged=true;
			}
	}//maxIter 	//52

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

//		printf("\ncpu sumWithin=%f", sumWithin);
//		printf("\ncpu sumInter=%f", sumInter);

		// Third objective function (Number of selected features)
		//pop[ind].fitness[2] = (float) nSelFeatures;
/* -- * /
					//CHECKS
		float auxSumaCentroids=0;	
		float auxSumaDist=0;
		int auxSumaSamples=0;
		int auxSumaMapping=0;
		int auxSumaNewMapping=0;

		for(int i=0; i<KMEANS*N_FEATURES; i++){
			auxSumaCentroids += centroids[i];
//			printf("\ncpu centroids[%d]= %f", i, centroids[i]);
		}
		for(int i=0; i<KMEANS*N_INSTANCES; i++){
			auxSumaDist += distCentroids[i];
//			printf("\ncpu distCentroids[%d]= %f", i, distCentroids[i]);
		}
		for(int i=0; i<KMEANS; i++){
			auxSumaSamples += samples_in_k[i];
//			printf("\ncpu samples_in_k[%d]= %d", i, samples_in_k[i]);
		}
		for(int i=0; i<host_tamMapping; i++){
			auxSumaMapping += mapping[i];
//			printf("\ncpu mapping[%d]=%d", i, mapping[i]);
		}
		for(int i=0; i<host_tamMapping; i++){
			auxSumaNewMapping += newMapping[i];
//			printf("\ncpu newMapping[%d]=%d", i, newMapping[i]);
		}

		printf("\nnVueltas=%d", nVueltas);


		printf("\nCPU: Suma total de centroids: %f", auxSumaCentroids);
		printf("\nCPU: Suma total de distCentroids: %f", auxSumaDist);
		printf("\nCPU: Suma total de samples_in_k: %d", auxSumaSamples);
		printf("\nCPU: Suma total de mapping: %d", 		auxSumaMapping);
		printf("\nCPU: Suma total de newMapping: %d", 	auxSumaNewMapping);
		printf("\nCPU Contadoorrr: %d", contadorr);
		printf("\nvalor de host_nextPowerTotalDistances= %d", host_nextPowerTotalDistances);
/* -- */
	}//for each individual
/* -- */	
	// Resources used are released
	free(mapping);
	free(newMapping);
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
void test_cpu_evaluation(individual *pop, const int begin, const int end, const float *const dataBase, const int nInstances, const int nFeatures, const unsigned char nObjectives, const int *const selInstances) {


	/************ Kmeans algorithm ***********/

	// Evaluate all the individuals and get the first and second objective for them
	test_cpu_kmeans(pop, begin, end, selInstances, dataBase);


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
void cpu_evaluation(individual *pop, const int begin, const int end, const float *const dataBase, const int nInstances, const int nFeatures, const unsigned char nObjectives, const int *const selInstances) {


	/************ Kmeans algorithm ***********/

	// Evaluate all the individuals and get the first and second objective for them
	cpu_kmeans(pop, begin, end, selInstances, dataBase);


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