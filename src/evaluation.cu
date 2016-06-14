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

/* -- * /
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
/* -- */

//nhebras = d_totalDistances
__global__ static 
void cuda_Convergence_Check(
//							float *dataBase, 
//							float *centroids,
//							unsigned char *member_chromosome,
							bool * mapping,
							bool * newMapping,
							bool * BlockConverged
							){
	const int d_totalDistances = KMEANS * N_INSTANCES;
	__shared__ bool  sharedMapping 		[d_totalDistances];
	__shared__ bool  sharedNewMapping 	[d_totalDistances];
	__shared__ bool  sharedConverged;

	unsigned int tx = threadIdx.x;
/* -- */
	sharedMapping[tx] = mapping[tx];
	sharedNewMapping[tx] = newMapping[tx];

	// Has the algorithm converged?
	if(tx==0){sharedConverged = true;}__syncthreads();	
/* -- */
	if(sharedNewMapping[tx] != sharedMapping[tx]){
		sharedConverged = false;
	}
/* -- */
	__syncthreads();
	if(tx==0){BlockConverged[blockIdx.x] = sharedConverged;}__syncthreads();
/* -- */


}

__global__ static
void cuda_Convergence_Euclidean(	
						float *dataBase, 
						float *centroids,
						unsigned char *member_chromosome,
						float * distCentroids,				
						bool * mapping,
						bool * newMapping,					
//						bool * auxMapping,
						int *samples_in_k
//						bool * converged,
//						int * nvueltas
					)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	int tx = threadIdx.x;
	const int d_totalDistances = KMEANS * N_INSTANCES;

	//extern __shared__ int sharedMemory[]; //main OPT point
//	__shared__ float sharedCentroids [KMEANS * N_FEATURES];
//	__shared__ unsigned char sharedMember_chromosome [N_FEATURES];
//	__shared__ float sharedDistCentroids[d_totalDistances];
//	__shared__ bool  sharedMapping 		[d_totalDistances];
//	__shared__ bool  sharedNewMapping 	[d_totalDistances];
//	__shared__ int sharedSamples_in_k [KMEANS];
	__shared__ int sharedThreadLater[N_INSTANCES];

	//Copiar valores a la memoria compartida;
	/* -- * /
	for(int i=threadIdx.x; i < tam; i+= blockDim.x){
		sharedCentroids[i] = centroids[i];
	}
	/* -- * /
	for(int i=threadIdx.x; i < N_FEATURES; i+= blockDim.x){
		sharedMember_chromosome[i] = member_chromosome[i];
	}
	/* -- * /
	for(int i=threadIdx.x; i < d_totalDistances; i+= blockDim.x){
		sharedDistCentroids[i] = distCentroids[i];
	}
	/* -- * /
	for(int i=threadIdx.x; i < d_totalDistances; i+= blockDim.x){
		sharedMapping[i] = mapping[i];
	}
	/* -- * /
	for(int i=threadIdx.x; i < d_totalDistances; i+= blockDim.x){
		sharedNewMapping[i]=false;
		sharedMapping[i]=false;
	}
	/* -- */
//	if(idx==0){
//		*(nvueltas)=0;
//		*converged= false;
//	}

//	for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && (!(*converged==true)); ++maxIter) {
		// The mapping table is cleaned in each iteration
	for(int i=threadIdx.x; i < d_totalDistances; i+= blockDim.x){
		newMapping[i] = false;
	}
	
	for(int i=threadIdx.x; i < KMEANS; i+= blockDim.x){
		samples_in_k[i] = 0;
	}
	__syncthreads();
	//A single thread executes the necessary work to compute an instance. Future improvements can me made. OPT
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

	if(idx==0){		//NO SE PUEDE PARALELIZAR ESTA MIERDA
		for(int i=0; i<N_INSTANCES; i++){
			samples_in_k[sharedThreadLater[i]]++;
		}
	}
/* -- */

/* -- * /
		}//max-iter
/* -- * /
	//Copiar los valores de salida
	if(idx < KMEANS * N_FEATURES){
		centroids[idx] = sharedCentroids[idx];
	}
	__syncthreads();
	if(idx < d_totalDistances){
		distCentroids[idx] = sharedDistCentroids[idx];
	}
	__syncthreads();	
	if(idx < d_totalDistances){
		mapping[idx]	= sharedMapping[idx];
	}
	__syncthreads();
	if(idx < d_totalDistances){
		newMapping[idx] = sharedNewMapping[idx];
	}
	__syncthreads();
	if(idx < KMEANS){
		samples_in_k[idx] = sharedSamples_in_k[idx];
	}
	__syncthreads();
/* -- */
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

	bool *mapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));
	bool *newMapping = (bool*) malloc(KMEANS * N_INSTANCES * sizeof(bool));

	const int totalDistances = KMEANS * N_INSTANCES;

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

		int *d_samples_in_k; 
		size_t size_6 = KMEANS * sizeof(int);
		checkCudaErrors(cudaMalloc((void **)&d_samples_in_k, size_6));	

		//d_converged is meant for internal use in cuda function
		bool * d_converged;
		checkCudaErrors(cudaMalloc(&d_converged, sizeof(bool)));

		int * d_nvueltas;
		checkCudaErrors(cudaMalloc(&d_nvueltas, sizeof(int)));

		bool * d_convergedThread;
		size_t size_7 = KMEANS * sizeof(bool);
		checkCudaErrors(cudaMalloc((void **)&d_convergedThread, size_7));

		//Capture individual chromosome
		unsigned char h_member_chromosome[N_FEATURES];
		for(int i=0; i<N_FEATURES; i++){
			h_member_chromosome[i] = pop[ind].chromosome[i];
		}

		cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

		//Se decide el nº de bloques
		const unsigned int numEuclideanThreadsPerBlock = N_INSTANCES;
		unsigned int numEuclideanBlocks = ((N_INSTANCES+numEuclideanThreadsPerBlock)-1) / deviceProp.maxThreadsPerMultiProcessor;
		if(numEuclideanBlocks==0){numEuclideanBlocks=1;}

		const unsigned int numConvergedThreadsPerBlock = totalDistances;
		unsigned int numConvergedBlocks = ((totalDistances+numConvergedThreadsPerBlock)-1) / deviceProp.maxThreadsPerMultiProcessor;
		if(numConvergedBlocks==0){numConvergedBlocks=1;}

		const unsigned int numAdjustThreadsPerBlock = N_FEATURES;
		unsigned int numAdjustBlocks = ((N_FEATURES+numAdjustThreadsPerBlock)-1) / deviceProp.maxThreadsPerMultiProcessor;
		if(numAdjustBlocks==0){numAdjustBlocks=1;}

		if(numEuclideanBlocks > (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount)){
			numEuclideanBlocks = (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		}	
//		if(numConvergedBlocks > (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount)){
//			numConvergedBlocks = (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
//		}
/* -- */
		long unsigned int BlockSharedEuclideanDataSize = KMEANS * N_INSTANCES * sizeof(bool) +		//sharedMapping
														 N_INSTANCES * sizeof(int)			+		//sharedThreadLater
														 0;
		long unsigned int BlockSharedConvergedDataSize = KMEANS * N_INSTANCES * sizeof(bool) +		//sharedMapping
														 KMEANS * N_INSTANCES * sizeof(bool) +		//sharedNewMapping
														 sizeof(bool) +								//sharedConverged
														 0;
/* -- */										

/* -- */

		if (BlockSharedConvergedDataSize > deviceProp.sharedMemPerBlock) {printf("WARNING: Your CUDA hardware has insufficient block shared memory.\n");}
//		if (BlockSharedEuclideanDataSize > deviceProp.sharedMemPerBlock) {printf("WARNING: Your CUDA hardware has insufficient block shared memory.\n");}

		/*		----------------------------------TODO:----------------------------------
				 calculate the amount of global memory necessary for the program to execute. Use this:
				        char msg[256];
        SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);
        		----------------------------------TODO:----------------------------------
        */


		bool * d_BlockConverged;
		size_t size_8 = numConvergedBlocks * sizeof(bool);
		checkCudaErrors(cudaMalloc((void **)&d_BlockConverged, size_8));



/* -- */
		//Copy values in device memory-------------------------------------
		checkCudaErrors(cudaMemcpy(d_dataBase, dataBase, size_1, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_centroids, centroids, size_2, cudaMemcpyHostToDevice));
//		checkCudaErrors(cudaMemcpy(d_mapping, mapping, size_5, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_member_chromosome, h_member_chromosome, size_3, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(d_distCentroids, distCentroids, size_4, cudaMemcpyHostToDevice));
		
		//This variables are initialized within the device kernel
		//checkCudaErrors(cudaMemcpy(d_newMapping, newMapping, size_5, cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMemcpy(d_samples_in_k, samples_in_k, size_6, cudaMemcpyHostToDevice));
/* -- */
		printf("\nNecesidad de hebras para Euclidean: %d", numEuclideanThreadsPerBlock);
		printf("\nNecesidad de hebras para Converged: %d", numConvergedThreadsPerBlock);
		printf("\nNumEuclideanBlocks: %d", numEuclideanBlocks);
		printf("\nNumConvergedBlocks: %d", numConvergedBlocks);
		printf("\n Numerajo: %d", (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount));
		printf("\n Euclidean - Se van a lanzar %d hebras repartidas en %d bloques", numEuclideanThreadsPerBlock*numEuclideanBlocks, numEuclideanBlocks);
		printf("\n Converged - Se van a lanzar %d hebras repartidas en %d bloques", numConvergedThreadsPerBlock*numConvergedBlocks, numConvergedBlocks);
		printf("\nUso de memoria compartida: %lu bytes / %lu", BlockSharedConvergedDataSize, deviceProp.sharedMemPerBlock);
		printf("\nUso de memoria compartida: %lu bytes / %lu", BlockSharedEuclideanDataSize, deviceProp.sharedMemPerBlock);



		/******************** Convergence process *********************/

		// Initialize the array of minimum distances and the mapping table
		float distCentroids[KMEANS * N_INSTANCES];
		int samples_in_k[KMEANS];

		// Initialize the mapping table
		for (int i = 0; i < totalDistances; ++i) {
			mapping[i] = false;
		}
		checkCudaErrors(cudaMemcpy(d_mapping, mapping, size_5, cudaMemcpyHostToDevice));
/* -- */
		int nVueltas=0;
		// To avoid poor performance, up to "MAX_ITER_KMEANS" iterations are executed
		bool converged = false;
//		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !converged; ++maxIter) {
			// The mapping table is cleaned in each iteration
			for (int i = 0; i < totalDistances; ++i) {
				newMapping[i] = false;
			}
			for (int i = 0; i < KMEANS; ++i) {
				samples_in_k[i] = 0;
			}
/* -- */
//			printf("\n -----[%d]----- Parte secuencial: calcular distancias Euclídeas", maxIter);
			// Calculate all distances (Euclidean distance) between each instance and the centroids
			for (int i = 0; i < N_INSTANCES; ++i) {
/* -- */
				float minDist = INFINITY;
				int selectCentroid = -1;
				int pos = N_FEATURES * i;
				for (int k = 0; k < KMEANS; ++k) {  //Para cada centroide
					float sum = 0.0f;
					int posCentroids = k * N_FEATURES;		//Nos situamos en el centroide
					int posDistCentr = k * N_INSTANCES;		//Nos situamos en [	i1 i2 i3 i4 i5 i6 i7 ]
					for (int f = 0; f < N_FEATURES; ++f) {	//					centroide 3
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
/*  -- */
//			printf("\n -----[%d]----- Parte secuencial: ver si ha convergido o no", maxIter);
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
/* -- */
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
						}//if chromosome
					}//for nfeatures
				}//if KMEANS
/* -- */
//				printf("\n -----[%d]----- Parte secuencial: intercambiar mappings", maxIter);
				// Swap mapping tables
				bool *aux = newMapping;
				newMapping = mapping;
				mapping = aux;
/* -- */
			}//!converged
//			printf("\n -----[%d]----- Parte secuencial: Finalizó MAXITER", maxIter);	


			nVueltas++;
/* -- * /
		}//maxiter----------------------------------------------------------------------------
/* -- */
		int gpu_nVueltas=0;
		bool gpu_converged = false;
//		for (int maxIter = 0; maxIter < MAX_ITER_KMEANS && !gpu_converged; ++maxIter) {
			cuda_Convergence_Euclidean <<< numEuclideanBlocks, numEuclideanThreadsPerBlock /*, BlockSharedConvergedDataSize */ >>> (	
																		d_dataBase, 
																		d_centroids, 
																		d_member_chromosome, 
																		d_distCentroids, 
																		d_mapping, 
																		d_newMapping,
																		d_samples_in_k
																		);
			cudaDeviceSynchronize();

			float gpu_distCentroids[KMEANS * N_INSTANCES];
			bool gpu_newMapping[KMEANS * N_INSTANCES];
			bool gpu_mapping[KMEANS * N_INSTANCES];
			int gpu_samples_in_k[KMEANS];
			float gpu_centroids[KMEANS * N_FEATURES];
/* -- */
			checkCudaErrors(cudaMemcpy(gpu_distCentroids, d_distCentroids, size_4, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_newMapping, d_newMapping, size_5, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_mapping, d_mapping, size_5, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_samples_in_k, d_samples_in_k, size_6, cudaMemcpyDeviceToHost));
//			checkCudaErrors(cudaMemcpy(gpu_centroids, d_centroids, size_2, cudaMemcpyDeviceToHost));
//			checkCudaErrors(cudaMemcpy(&gpu_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
//			checkCudaErrors(cudaMemcpy(&gpu_nvueltas, d_nvueltas, sizeof(int), cudaMemcpyDeviceToHost));
//			checkCudaErrors(cudaMemcpy(gpu_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
//			cudaMemcpyFromSymbol(&gpu_converged, "CUDAconverged", sizeof(bool), 0, cudaMemcpyDeviceToHost);
			//**********************************************************************************************
/* -- */
			//Comprobación de si los resultados son iguales que en secuencial:
			printf("\nComprobando CONVERGENCE EUCLIDEAN---------------------\n");

			int tam_mapping = KMEANS * N_INSTANCES;
			unsigned int numDist=0;
/* -CHECK- */
			for(int i=0; i< tam_mapping; i++){
				if(gpu_distCentroids[i] != distCentroids[i]){
					numDist++;
					//printf("\ndistCentroids[%d] no encaja con la versión secuencial.\n", i);
					//printf("%f\n%f\n", distCentroids[i], gpu_distCentroids[i]);
				}else{
					//printf("\ndistCentroids[%d] encaja con la versión secuencial.\n", i);
				}
			}
			printf("DistCentroids 'mal': %d/%d \n", numDist, tam_mapping);
/* -CHECK- */
			printf("\nSamples:\n");
			for(int i=0; i<KMEANS; i++){
				if(gpu_samples_in_k[i] != samples_in_k[i]){
					printf("\nsamples_in_k[%d] no encaja con la versión secuencial.\n", i);
					printf("[%d]   %d %d\n", i, samples_in_k[i], gpu_samples_in_k[i]);
				}
			}
/* -CHECK- * /
			printf("\nCentroids:\n");
			for(int i=0; i< KMEANS * N_FEATURES; i++){
				if(gpu_centroids[i] != centroids[i]){
					printf("\ncentroids[%d] no encaja con la versión secuencial.\n", i);
					printf("%f\n%f\n", centroids[i], gpu_centroids[i]);
				}else{
					//printf("\ncentroids[%d] encaja con la versión secuencial.\n", i);
				}
			}
/* -CHECK- */
			printf("\nmappings normales y de la gpu:\n");
			for(int i=0; i<tam_mapping; i++){
				if( (mapping[i] !=   gpu_newMapping[i]   ) 
								||
					(newMapping[i] != gpu_mapping[i])  
															){
					if(i<10){
						printf("[%d]   %d %d\n      %d %d\n\n", i, mapping[i], gpu_mapping[i], newMapping[i], gpu_newMapping[i]);
					}else if(i<100){
						printf("[%d]   %d %d\n       %d %d\n\n", i, mapping[i], gpu_mapping[i], newMapping[i], gpu_newMapping[i]);
					}else{
						printf("[%d]   %d %d\n        %d %d\n\n", i, mapping[i], gpu_mapping[i], newMapping[i], gpu_newMapping[i]);
					}
				}
			}
/* -CHECK- */

		cuda_Convergence_Check <<< numConvergedBlocks, numConvergedThreadsPerBlock /*, BlockSharedEuclideanDataSize */ >>> (	
															d_mapping,
															d_newMapping,
															d_BlockConverged
															);
		cudaDeviceSynchronize();
		bool gpu_BlockConverged[numConvergedBlocks];

		checkCudaErrors(cudaMemcpy(gpu_BlockConverged, d_BlockConverged, size_8, cudaMemcpyDeviceToHost));
		//gpu_BlockConverged[0]=false;
		printf("\nComprobando CONVERGENCE CHECK---------------------\n");
		gpu_converged = true;
		for(int i=0; i < numConvergedBlocks; i++){
			if(!gpu_BlockConverged[i]){
				gpu_converged = false;
			}
			printf("\nConverged en gpu: %d\n", gpu_BlockConverged[i]);
		}
		printf("\nConverged en seq: %d\n", converged);

		if(!gpu_converged){

			checkCudaErrors(cudaMemcpy(gpu_centroids, d_centroids, size_2, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_mapping, d_mapping, size_5, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_newMapping, d_newMapping, size_5, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(gpu_samples_in_k, d_samples_in_k, size_6, cudaMemcpyDeviceToHost));
			
			//HAY QUE HASERLO CON LA CPU VIEOH
			// Update the position of the centroids
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

						gpu_centroids[posCentroids + f] = sum / gpu_samples_in_k[k];
					}//if chromosome
				}//for nfeatures
			}//for KMEANS
	/* -- */
			// Swap mapping tables
			checkCudaErrors(cudaMemcpy(d_mapping, newMapping, size_5, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_newMapping, mapping, size_5, cudaMemcpyHostToDevice));
			//New centroids, thanks to CPU work
			checkCudaErrors(cudaMemcpy(d_centroids, centroids, size_2, cudaMemcpyHostToDevice));
		}//!converged
/* -CHECK- */
		printf("\nComprobando CONVERGENCE AJUSTAR---------------------\n");
			printf("\nmappings normales y de la gpu:\n");
			for(int i=0; i<tam_mapping; i++){
				if( (mapping[i] !=  gpu_newMapping[i]) 
								||
					(newMapping[i] != gpu_mapping[i])  
															){
					if(i<10){
						printf("[%d]   %d %d\n      %d %d\n\n", i, mapping[i], gpu_mapping[i], newMapping[i], gpu_newMapping[i]);
					}else if(i<100){
						printf("[%d]   %d %d\n       %d %d\n\n", i, mapping[i], gpu_mapping[i], newMapping[i], gpu_newMapping[i]);
					}else{
						printf("[%d]   %d %d\n        %d %d\n\n", i, mapping[i], gpu_mapping[i], newMapping[i], gpu_newMapping[i]);
					}
				}
			}
/* -CHECK- */			
			printf("\nCentroids:\n");
			for(int i=0; i< KMEANS * N_FEATURES; i++){
				if(gpu_centroids[i] != centroids[i]){
					printf("\ncentroids[%d] no encaja con la versión secuencial.\n", i);
					printf("%f\n%f\n", centroids[i], gpu_centroids[i]);
				}else{
					//printf("\ncentroids[%d] encaja con la versión secuencial.\n", i);
				}
			}
/* -CHECK- */
			gpu_nVueltas++;
//		}//maxIter

		printf("\nN vueltas seq: %d y en gpu: %d", nVueltas, gpu_nVueltas);
		printf("\n ---------- Parte secuencial: Empezando WCSS y ICSS");
		/************ Minimize the within-cluster and maximize Inter-cluster sum of squares (WCSS and ICSS) *************/

/* -- * /
		int totalCoord = KMEANS * N_FEATURES;
		float sumWithin = 0.0f;
		float sumInter = 0.0f;
//		for(int i=0; i < totalCoord; i++){
//			sumasParciales[i]=0;
//		}

		for (int k = 0; k < KMEANS; ++k) {
//			printf("\n\nValor de k: %d", k);
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
//				printf("\nValor de i: %d", i);
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
		printf("\n ---------- Parte secuencial: Terminado WCSS y ICSS");

		checkCudaErrors(cudaFree(d_dataBase));  
		checkCudaErrors(cudaFree(d_centroids));
		checkCudaErrors(cudaFree(d_member_chromosome));
		checkCudaErrors(cudaFree(d_distCentroids));
		checkCudaErrors(cudaFree(d_mapping));
		checkCudaErrors(cudaFree(d_newMapping));
		checkCudaErrors(cudaFree(d_auxMapping));
		checkCudaErrors(cudaFree(d_samples_in_k));
//		checkCudaErrors(cudaFree(d_converged));
		checkCudaErrors(cudaFree(d_convergedThread));
		checkCudaErrors(cudaFree(d_nvueltas));
//		checkCudaErrors(cudaFree(d_result_sumWithin));
//		checkCudaErrors(cudaFree(d_result_sumInter));
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

/* -- * /
__global__ static
void cuda_WCSSICSS( 
						float *centroids,					//OK
						unsigned char *member_chromosome,	//OK
						float * distCentroids,				//OK
						bool * mapping,						//OK
						float * result_sumWithin,
						float * result_sumInter
					)
{
	const int d_totalDistances = KMEANS * N_INSTANCES;
	int tx = threadIdx.x;

	__shared__ float sharedCentroids [KMEANS * N_FEATURES];
	__shared__ unsigned char sharedMember_chromosome [N_FEATURES];		
	__shared__ float sharedDistCentroids[d_totalDistances];
	__shared__ bool  sharedMapping 		[d_totalDistances];
	__shared__ float sharedResult_sumWithin;
	__shared__ float sharedResult_sumInter;

	
	
	//Copiar valores a la memoria compartida;
	if(tx < KMEANS * N_FEATURES){
		sharedCentroids[tx] = centroids[tx];
	}
	__syncthreads();
	if(tx < N_FEATURES){
		sharedMember_chromosome[tx] = member_chromosome[tx];
	}
	__syncthreads();
	if(tx < d_totalDistances){
		sharedDistCentroids[tx] = distCentroids[tx];
	}
	__syncthreads();
	if(tx < d_totalDistances){
		sharedMapping[tx] = mapping[tx];
	}

	__shared__ int totalCoord;
	__shared__ float sumWithin[KMEANS];
	__shared__ float sumInter[KMEANS];
//		for (int k = 0; k < KMEANS; ++k) {
	__syncthreads();


	if(tx == 0){
		totalCoord = KMEANS * N_FEATURES;
	}

	if(tx < KMEANS){
		int posCentroids = tx * N_FEATURES;
		int posDistCentr = tx * N_INSTANCES;

//		sumWithin[tx] = 0.0f;
//		sumInter[tx] = 0.0f;

		// Within-cluster
		for (int i = 0; i < N_INSTANCES; ++i) {
			if (sharedMapping[posDistCentr + i]) {
				sharedResult_sumWithin = sharedResult_sumWithin + sharedDistCentroids[posDistCentr + i];
			}
		}

		// Inter-cluster
		for (int i = posCentroids + N_FEATURES; i < totalCoord; i += N_FEATURES) {
			float sum = 0.0f;
			for (int f = 0; f < N_FEATURES; ++f) {
				if (sharedMember_chromosome[f] & 1) {
					sum += (sharedCentroids[posCentroids + f] - sharedCentroids[i + f]) * (sharedCentroids[posCentroids + f] - sharedCentroids[i + f]);
				}
			}
			__syncthreads();
			sharedResult_sumInter += sqrt(sum);
			__syncthreads();
		}
	}//WCSS and ICSS minimization process
/* -- * /	
	__syncthreads();
//	if(tx < KMEANS){						//TODO: make a proper reduction operation
	if(tx == 0){
		sharedResult_sumWithin = 0;
		sharedResult_sumInter = 0;
		for(int i=0; i < KMEANS; i++){
			sharedResult_sumWithin += sumWithin[tx];
			sharedResult_sumInter  += sumInter[tx];
		}
	}
	__syncthreads();
/* -- * /	
	if(tx==0){
//		*(result_sumWithin) = 5.0;
//		*(result_sumInter) = 5.0;
		*(result_sumWithin) = sharedResult_sumWithin;
		*(result_sumInter) = sharedResult_sumInter;

	}
/* -- * /


}
/* -- */