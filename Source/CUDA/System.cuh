extern "C"
{

//  utility, data
void cudaInit(int argc, char **argv);
bool cudaInitB(int argc, char **argv, bool showInfo);
void threadSync();

void allocateArray(void **devPtr, int size);
void freeArray(void *devPtr);

void copyFromDevice(void* host, const void* device, struct cudaGraphicsResource** ,int size);
void copyToDevice(void* device, const void* host, int offset, int size);

//void registerGLvbo(uint vbo);
//void unregGLvbo(uint vbo);

void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
void unmapGLBufferObjects(int cout,struct cudaGraphicsResource *cuda_vbo_resource);

void setParameters(SimParams *hostParams);


//  System

void integrate(float4* oldPos, float4* oldVel, float4* newPos, float4* newVel,
		int numParticles);

void calcHash(float4* pos, uint* particleHash, uint* particleIndex, uint numParticles);

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

void reorder(float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel,
		uint* particleHash, uint* particleIndex, uint* cellStart, uint* cellEnd, uint numParticles, uint numCells);

void collide(float4* newVel, float4* vboCLR,
		float4* sortedPos, float4* sortedVel,  
		float* pressure, float* density, float* dyeColor,//
		uint* particleHash,uint* particleIndex, uint* cellStart, uint* cellEnd, uint numParticles, uint numCells);


}
