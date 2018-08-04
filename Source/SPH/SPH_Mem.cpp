#include "header.h"

#include "..\SPH\SPH.h"
#include "..\CUDA\System.cuh"
#include "..\CUDA\radixsort.cuh"



///  Init data, allocate memory
///---------------------------------------------------------------------------------------------------
void cSPH::_InitMem()
{
	if (bInitialized)  return;	bInitialized = true;

	//  CPU data
	uint sflt = sizeof(float), sui = sizeof(uint),
		npar = scn.params.numParticles, ngc = scn.params.numCells,
		memSize1 = npar*sflt, memSize4 = memSize1*4;
	
	hPos = new float4[npar];		memset(hPos, 0, memSize4);
	hVel = new float4[npar];		memset(hVel, 0, memSize4);
	hParHash = new uint[npar*2];	memset(hParHash, 0, npar*2 *sui);
	hCellStart	= new uint[ngc];	memset(hCellStart, 0, ngc *sui);
	//hCounters	= new int[10];		memset(hCounters,  0, 10 *sui);
	m_hCellStart = new uint[scn.params.numCells];
    memset(m_hCellStart, 0, scn.params.numCells*sizeof(uint));

	//  GPU data
	posVbo[0] = createVBO(memSize4);	posVbo[1] = createVBO(memSize4);
	registerGLBufferObject(posVbo[0], &m_cuda_posvbo_resource[0]);  
	registerGLBufferObject(posVbo[1], &m_cuda_posvbo_resource[1]);
	
	allocateArray((void**)&dVel[0], memSize4);		allocateArray((void**)&dVel[1], memSize4);
	allocateArray((void**)&dSortedPos, memSize4);	allocateArray((void**)&dSortedVel, memSize4);
	allocateArray((void**)&dPressure, memSize1);	allocateArray((void**)&dDensity, memSize1);
	allocateArray((void**)&dDyeColor, memSize1);//

	allocateArray((void**)&dParHash, npar *sui);	
	allocateArray((void**)&dParIndex, npar *sui);
	allocateArray((void**)&dCellStart, ngc *sui);
	allocateArray((void**)&dCellEnd, ngc *sui);
	//allocateArray((void**)&dCounters[0], 100 *sui);		allocateArray((void**)&dCounters[1], 100 *sui);

	colorVbo = createVBO(memSize4);
	registerGLBufferObject(colorVbo, &m_cuda_colorvbo_resource);

	//fill color buffer 
	glBindBufferARB(GL_ARRAY_BUFFER, colorVbo);
	float *data = (float*) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;
	for(uint i=0; i < npar; i++)  {
		float t = i / (float)npar;
		colorRamp(t, ptr);	ptr+=3;  *ptr++ = 1.0f;  }
	glUnmapBufferARB(GL_ARRAY_BUFFER);/**/

	setParameters(&scn.params);
}


//  free memory
//----------------------------------------------------------------------------------------------------
void cSPH::_FreeMem()
{
	if (!bInitialized)  return;  bInitialized = false;
	
	#define  DELA(a)	if (a)	{	delete[] a;  a = NULL;	}

	DELA(hPos);  DELA(hVel);
	DELA(hParHash);  DELA(hCellStart);
	//DELA(hCounters);
	
	freeArray(dVel[0]);		freeArray(dVel[1]);
	freeArray(dSortedPos);	freeArray(dSortedVel);
	freeArray(dPressure);	freeArray(dDensity);
	freeArray(dDyeColor);//

	freeArray(dParHash);  
	freeArray(dParIndex);
	freeArray(dCellStart);
	freeArray(dCellEnd);
	//freeArray(dCounters[0]);  freeArray(dCounters[1]);

	//unregGLvbo(posVbo[0]);	unregGLvbo(posVbo[1]);
	//unregGLvbo(colorVbo);

	unregisterGLBufferObject(m_cuda_posvbo_resource[0]);
	unregisterGLBufferObject(m_cuda_posvbo_resource[1]);
	glDeleteBuffers(2, (const GLuint*)posVbo);
	glDeleteBuffers(1, (const GLuint*)&colorVbo);
}

