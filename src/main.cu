/**
 * @file main.cpp
 * @author Miguel Sánchez Tello
 * @date 26/06/2016
 * @brief Multiobjective genetic algorithm
 *
 * Multiobjective genetic algorithm running on a general purpose processor
 *
 */

const int BLOCK_SIZE=1024;

/********************************* Includes ********************************/

#include "tinyxml2.h"
#include "xml.h"
#include "bd.h"
#include "initialization.h"
#include "evaluation.h"
#include "sort.h"
#include "tournament.h"
#include "crossover.h"
#include <stdio.h> // fprintf...
#include <time.h> // clock...

#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <curand.h>
#include <curand_kernel.h>


using namespace tinyxml2;


/**
 * @brief Main program
 * @param argc The number of arguments of the program
 * @param argv Arguments of the program
 * @return Returns nothing if successful or a negative number if incorrect
 */
int main(int argc, char** argv) {


	/********** Get the configuration data from the XML file ***********/
	XMLDocument configDoc;
	configDoc.LoadFile(argv[1]);
	const char *dataBaseName = getDataBaseName(&configDoc);
	const int nGenerations = getNGenerations(&configDoc);
	const int maxFeatures = getMaxFeatures(&configDoc);
	const int tourSize = getTourSize(&configDoc);
	const int crossDistribution = getCrossDistribution(&configDoc);
	const int mutDistribution = getMutDistribution(&configDoc);
	const char *dataName = getDataName(&configDoc);
	const char *plotName = getPlotName(&configDoc);
	const char *imageName = getImageName(&configDoc);

	/********** Check program restrictions ***********/

	if (POPULATION_SIZE < 4) {
		fprintf(stderr, "Error: The number of individuals must be 4 or higher\n");
		exit(-1);
	}

	if (N_FEATURES < 2 || N_INSTANCES < 2) {
		fprintf(stderr, "Error: The number of features and number of instances must be 2 or higher\n");
		exit(-1);
	}

	if (N_OBJECTIVES != 2) {
		fprintf(stderr, "Error: The number of objectives must be 2. If you want to increase this number, the module \"evaluation\" should be modified\n");
		exit(-1);
	}

	if (maxFeatures < 1) {
		fprintf(stderr, "Error: The maximum initial number of features must be 1 or higher\n");
		exit(-1);
	}

	if (tourSize < 2) {
		fprintf(stderr, "Error: The number of individuals in the tournament must be 2 or higher\n");
		exit(-1);
	}

	if (crossDistribution < 0 || mutDistribution < 0) {
		fprintf(stderr, "Error: The cross distribution and the mutation distribution must be 0 or higher\n");
		exit(-1);
	}

/* ----------------------------- */
    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
        SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (deviceCount >= 2)
    {
        cudaDeviceProp prop[64];
        int gpuid[64]; // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;

        for (int i=0; i < deviceCount; i++)
        {
            checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled to support this
                && prop[i].tccDriver
#endif
               )
            {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;

        if (gpu_p2p_count >= 2)
        {
            for (int i = 0; i < gpu_p2p_count; i++)
            {
                for (int j = 0; j < gpu_p2p_count; j++)
                {
                    if (gpuid[i] == gpuid[j])
                    {
                        continue;
                    }
                    checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                        printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid[i]].name, gpuid[i],
                           prop[gpuid[j]].name, gpuid[j] ,
                           can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;

    // Print Out all device Names
    for (dev = 0; dev < deviceCount; ++dev)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
        sprintf(cTemp, ", Device%d = ", dev);
#endif
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += cTemp;
        sProfileString += deviceProp.name;
    }

    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

    printf("Result = PASS\n");
/* ----------------------------- */

    const int totalDistances = KMEANS * N_INSTANCES;
    unsigned int numEuclideanThreadsPerBlock = BLOCK_SIZE;
    unsigned int numEuclideanBlocks = ((N_INSTANCES+numEuclideanThreadsPerBlock)-1) / numEuclideanThreadsPerBlock;
    unsigned int numWithinThreadsPerBlock = BLOCK_SIZE;
    unsigned int numWithinBlocks = ((totalDistances+totalDistances)-1) / BLOCK_SIZE;
    printf("\n Euclidean - Se van a lanzar %d hebras repartidas en %d bloques.", numEuclideanThreadsPerBlock*numEuclideanBlocks, numEuclideanBlocks);
    printf("\n Within    - Se van a lanzar %d hebras repartidas en %d bloques.", numWithinBlocks * numWithinThreadsPerBlock, numWithinBlocks);

    float CUDAtime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    clock_t t_ini, t_fin;
    t_ini = clock();
	/********** Get the data base ***********/

	float h_dataBase[N_INSTANCES * N_FEATURES];

	readDataBase(h_dataBase, dataBaseName, N_INSTANCES, N_FEATURES);

	// Data base normalization
	normDataBase(h_dataBase, N_INSTANCES, N_FEATURES);
	


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for data base reading and normalization:  %3.1f ms", CUDAtime);
//	t_checkpoint1 = clock();
//	ms = ((double) (t_checkpoint1 - t_ini) / CLOCKS_PER_SEC) * 1000.0;
//	fprintf(stdout, "Time for data base reading and normalization: %.16g\n", ms);

	cudaEventRecord(start, 0);
	/********** Initialize the population and the individuals ***********/

	srand((unsigned int) time(NULL));
	const int totalIndividuals = POPULATION_SIZE << 1;

	// Population will have the parents and children (left half and right half respectively
	// This way is better for the performance
	individual *population = initPopulation(totalIndividuals, N_OBJECTIVES, N_FEATURES, maxFeatures);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for initialization of population:  %3.1f ms", CUDAtime);


	/********** Multiobjective individual evaluation ***********/

	// Get the initial "KMEANS" centroids *********** /
	int selInstances[KMEANS];
	getCentroids(selInstances, N_INSTANCES);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for multiobjective individual evaluation - Centroids:  %3.1f ms", CUDAtime);

/* -- * /
	evaluation(population, 0, POPULATION_SIZE, h_dataBase, N_INSTANCES, N_FEATURES, N_OBJECTIVES, selInstances);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for multiobjective individual evaluation - Evaluation:  %3.1f ms", CUDAtime);

/* -- */
//	CUDA_evaluation(population, 0, POPULATION_SIZE, h_dataBase, N_INSTANCES, N_FEATURES, N_OBJECTIVES, selInstances);
	CUDA_evaluation(population, 0, POPULATION_SIZE, h_dataBase, N_INSTANCES, N_FEATURES, N_OBJECTIVES, selInstances);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for multiobjective individual evaluation - Evaluation:  %3.1f ms", CUDAtime);
/* -- */

	/********** Sort the population with the "Non-Domination-Sort" method ***********/

	int nIndFront0 = nonDominationSort(population, POPULATION_SIZE, N_OBJECTIVES, N_INSTANCES, N_FEATURES);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for sort the population with Non-Domination-Sort:  %3.1f ms", CUDAtime);


	/********** Get the population quality (calculating the hypervolume) ***********/

	// The reference point will be (X_1 = 1.0, X_2 = 1.0, .... X_N_OBJECTIVES = 1.0)
	double referencePoint[N_OBJECTIVES];
	for (int i = 0; i < N_OBJECTIVES; ++i) {
		referencePoint[i] = 1.0;
	}

	float popHypervolume = getHypervolume(population, nIndFront0, N_OBJECTIVES, referencePoint);
	float auxHypervolume;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for hypervolume:  %3.1f ms", CUDAtime);


	/********** Start the evolution process ***********/

	const int poolSize = POPULATION_SIZE >> 1;
	int pool[poolSize];
	//for (int g = 0; g < nGenerations && popHypervolume > auxHypervolume; ++g) {
	for (int g = 0; g < nGenerations; ++g) {

		// Fill the mating pool
		fillPool(pool, poolSize, tourSize, POPULATION_SIZE);

		// Perform crossover
		int nChildren = crossover(population, POPULATION_SIZE, pool, poolSize, N_OBJECTIVES, N_FEATURES, crossDistribution, mutDistribution);

		// Multiobjective individual evaluation
		int lastChild = POPULATION_SIZE + nChildren;
//		evaluation(population, POPULATION_SIZE, lastChild, h_dataBase, N_INSTANCES, N_FEATURES, N_OBJECTIVES, selInstances);
        CUDA_evaluation(population, POPULATION_SIZE, lastChild, h_dataBase, N_INSTANCES, N_FEATURES, N_OBJECTIVES, selInstances);
		
		// The crowding distance of the parents is initialized again for the next nonDominationSort
		for (int i = 0;  i < POPULATION_SIZE; ++i) {
			population[i].crowding = 0.0f;
		}

		// Replace population
		// Parents and children are sorted by rank and crowding distance.
		// The first "populationSize" individuals will advance the next generation
		nIndFront0 = nonDominationSort(population, POPULATION_SIZE + nChildren, N_OBJECTIVES, N_INSTANCES, N_FEATURES);
		
		// Get the population quality (calculating the hypervolume)
		auxHypervolume = getHypervolume(population, nIndFront0, N_OBJECTIVES, referencePoint);
	}

	popHypervolume = auxHypervolume;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CUDAtime, start, stop);
	printf("\nTime for evolution process:  %3.1f ms\n", CUDAtime);

	// Finish the time measure
	t_fin = clock();
	double ms = ((double) (t_fin - t_ini) / CLOCKS_PER_SEC) * 1000.0;
	fprintf(stdout, "Total time: %.16g ms\n", ms);
	fprintf(stdout, "%f\n", popHypervolume);

	// Generation of the data file and Gnuplot file for display the Pareto front
	generateGnuplot(dataName, plotName, imageName, population, nIndFront0, N_OBJECTIVES, referencePoint);


	/********** Resources used are released ***********/

	// The individuals (parents and children)
	delete[] population;

    // finish
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    printf("\n");
}