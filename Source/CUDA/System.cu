#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

#include <helper_cuda.h>
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms 

#include "Kernel.cu"
#include "radixsort.cu"
#include "device_launch_parameters.h"

extern "C"
{
	 
	//  Utility, data

#define cuMapVbo(pv,vbo)		cudaGLMapBufferObject((void**)&pv, vbo);
#define cuUnMapVbo(vbo)			cudaGLUnmapBufferObject(vbo);

#define cuBindTex(tex,pv,size)	checkCudaErrors(cudaBindTexture(0, tex, pv,size));
#define cuUnbindTex(tex)		checkCudaErrors(cudaUnbindTexture(tex));
	// constants
	const unsigned int mesh_width    = 256;
	const unsigned int mesh_height   = 256;
	// vbo variables
	GLuint vbo;
	void *d_vbo_buffer = NULL;
	float g_fAnim = 0.0;
	// mouse controls
	int mouse_old_x, mouse_old_y;
	int mouse_buttons = 0;
	float rotate_x = 0.0, rotate_y = 0.0;
	float translate_z = -3.0;


	void cudaInit(int argc, const char **argv)	{   findCudaDevice(argc, argv);	}
	bool cudaInitB(int argc, char **argv, bool showInfo)	//  no exit + info
	{
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

		int dev = 0, driverVersion = 0, runtimeVersion = 0;

		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		if (showInfo)
		{
			// Console log
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);
			printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
			printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
			printf("Total memory:  %u bytes  (%u MB)\n", deviceProp.totalGlobalMem, deviceProp.totalGlobalMem/1024/1024);
#if CUDART_VERSION >= 5000
			// This is supported in CUDA 5.0 (runtime API device properties)
			printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
			printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);
			printf("  Warp size:				%d\n", deviceProp.warpSize);
			printf("  Constant memory:			%u bytes\n", deviceProp.totalConstMem); 
			printf("  Shared memory per block:  %u bytes\n", deviceProp.sharedMemPerBlock);
			printf("  Registers per block:   %d\n", deviceProp.regsPerBlock);
			printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
			printf("  Max sizes of a block:  %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
			printf("  Max sizes of a grid:   %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
			printf("  Max memory pitch:    %u bytes\n", deviceProp.memPitch);
			printf("  Texture alignment:   %u bytes\n", deviceProp.textureAlignment);

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
		}


		return true; 
	}

	void threadSync()						{	checkCudaErrors(cudaDeviceSynchronize());	}


	void allocateArray(void **devPtr, size_t size)	{	cudaMalloc(devPtr, size);	}
	void freeArray(void *devPtr)					{	cudaFree(devPtr);	}

	//void registerGLvbo(uint vbo)		{	cudaGLRegisterBufferObject(vbo);	}
	//void unregGLvbo(uint vbo)			{	cudaGLUnregisterBufferObject(vbo);	}
	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsNone));
	}

	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
	}

	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource) 
	{
		void *ptr;
		checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
			*cuda_vbo_resource));
		return ptr;
	}

	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

	void unmapGLBufferObjects(int count ,struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnmapResources(count, &cuda_vbo_resource, 0));
	}
	
	void copyToDevice(void* device, const void* host, int offset, int size)
	{
		checkCudaErrors(cudaMemcpy((char*) device + offset, host, size, cudaMemcpyHostToDevice));
	}

	void copyFromDevice(void* host, const void* device,  struct cudaGraphicsResource **cuda_vbo_resource, int size)
	{
		//if (vbo)  cuMapVbo(device, vbo);
		if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }
		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
		//if (vbo)  cuUnMapVbo(vbo);
		if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
	}

	void setParameters(SimParams *hostParams)
	{	// copy parameters to constant memory
		checkCudaErrors(cudaMemcpyToSymbol(par, hostParams, sizeof(SimParams)));
	}
	 
	 
	//  Round a / b to nearest higher integer value
	int iDivUp(int a, int b) {	return a%b != 0 ? a/b+1 : a/b;	}

	//  compute grid and thread block size for a given number of elements
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
	{
		numThreads = min(blockSize, n);  numBlocks = iDivUp(n, numThreads);
	}
	//----------------------------------------------------------------------------------




///----------------------------------------------------------------------------------------------------------------------------
///  Integrate
///----------------------------------------------------------------------------------------------------------------------------

	void integrate( float4* oldPos, float4* oldVel, float4* newPos,  float4* newVel,uint numParticles)
	{
		uint numThreads, numBlocks;	
		computeGridSize(numParticles, 256, numBlocks, numThreads);
		
		integrateD<<< numBlocks, numThreads >>>(newPos, newVel, oldPos, oldVel); 
	} 

	///  calcHash
	void calcHash(float4* pos, uint* particleHash, uint* particleIndex, uint numParticles)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		//float4 *pos;
		//cuMapVbo(pos, vboPos);
		calcHashD<<< numBlocks, numThreads >>>(pos, particleHash, particleIndex, numParticles);
		getLastCudaError("Kernel execution failed");
		//cuUnMapVbo(vboPos); //move to cSPH::Update()
		
	}

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

	///  reorder
	void reorder(float4 *oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel,
		uint* particleHash, uint* particleIndex, uint* cellStart, uint* dCellEnd, uint numParticles, uint numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);
		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));
		#if USE_TEX
		uint spar4 = numParticles*sizeof(float4);
		cuBindTex(oldPosTex, oldPos, spar4);	cuBindTex(oldVelTex, oldVel, spar4);
		#endif
		//float4 *oldPos; 
		//checkCudaErrors(cudaGLMapBufferObject((void**)&oldPos, vboOldPos));
		reorderD<<< numBlocks, numThreads >>>(particleHash, particleIndex, cellStart, dCellEnd,
			oldPos, oldVel, sortedPos, sortedVel, numParticles);
		getLastCudaError("Kernel execution failed: reorderD");
		#if USE_TEX 
		cuUnbindTex(oldPosTex);  cuUnbindTex(oldVelTex);
		#endif
		//cuUnMapVbo(vboOldPos);
		
	 }
	  

	///  collide 
	void collide( float4* newVel, float4* clr2,
		float4* sortedPos, float4* sortedVel, 
		float* pressure, float* density, float* dyeColor, 
		uint* particleHash, uint* particleIndex, uint* cellStart, uint* cellEnd, uint numParticles, uint numCells)
	{ 
		#if USE_TEX
		uint spar4 = numParticles*sizeof(float4), spar = numParticles*sizeof(float);
		cuBindTex(oldPosTex, sortedPos, spar4);	cuBindTex(oldVelTex, sortedVel, spar4);
		cuBindTex(pressureTex, pressure, spar); cuBindTex(densityTex, density, spar); cuBindTex(dyeColorTex, dyeColor, spar);
		cuBindTex(cellStartTex, cellStart, numCells*sizeof(uint));
		cuBindTex(cellEndTex, cellEnd, numCells*sizeof(uint));
		#endif
		
		uint numThreads, numBlocks; 
		computeGridSize(numParticles, 64, numBlocks, numThreads);  
		
		computeDensityD<<< numBlocks, numThreads >>>( sortedPos, pressure, density, particleHash, cellStart, cellEnd);  
		  
		computeForceD<<< numBlocks, numThreads >>>( newVel, sortedPos, sortedVel, clr2, pressure, density, dyeColor/**/, particleHash, particleIndex, cellStart, cellEnd);
		 
		#if USE_TEX 
		cuUnbindTex(oldPosTex);		cuUnbindTex(oldVelTex);
		cuUnbindTex(pressureTex)	cuUnbindTex(densityTex);	cuUnbindTex(dyeColorTex);//
		cuUnbindTex(cellStartTex);  cuUnbindTex(cellEndTex);   
		#endif
	} 

}  //extern "C"