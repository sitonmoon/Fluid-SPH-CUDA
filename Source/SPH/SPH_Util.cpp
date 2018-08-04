#include "header.h"

#include "..\SPH\SPH.h"
#include "..\CUDA\System.cuh"



uint cSPH::createVBO(uint size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//registerGLvbo(vbo);//Deprecated
	return vbo;
}


inline float lerp(float a, float b, float t)  {   return a + t*(b-a);   }

void cSPH::colorRamp(float t, float *r)
{
	const int ncolors = 4;
	float c[ncolors][3] = {
		{ 0.6f, 0.8f, 1.0f, },
		{ 0.2f, 0.5f, 1.0f, },
		{ 0.5f, 0.2f, 1.0f, },
		{ 1.0f, 1.0f, 1.0f, },  };
	
	t = t * (ncolors-1);
	int i = (int) t;
	float u = t - floor(t);
	
	r[0] = lerp(c[i][0], c[i+1][0], u);
	r[1] = lerp(c[i][1], c[i+1][1], u);
	r[2] = lerp(c[i][2], c[i+1][2], u);
}



float* cSPH::getArray(bool pos)
{
	assert(bInitialized);
 
	float* hdata = 0, *ddata = 0;	//uint vbo = 0;
	struct cudaGraphicsResource *cuda_vbo_resource = 0;

	//if (!pos)
	//{	hdata = hPos;  ddata = dPos[curPosRead];	//vbo = posVbo[curPosRead];
	//	cuda_vbo_resource = m_cuda_posvbo_resource_read;	}
	//else
	//{	hdata = hVel;  ddata = dVel[curVelRead];	}

	copyFromDevice(hdata, ddata, &cuda_vbo_resource, scn.params.numParticles*4*sizeof(float));
	return hdata;
}


void cSPH::setArray(bool pos, const float4* data, int start, int count)
{
	assert(bInitialized);
	const uint si4 = 4*sizeof(float);
	if (!pos)
	{	//unregGLvbo(posVbo[curPosRead]);
		unregisterGLBufferObject(m_cuda_posvbo_resource[curPosRead]);
		glBindBuffer(GL_ARRAY_BUFFER, posVbo[curPosRead]);
		glBufferSubData(GL_ARRAY_BUFFER, start*si4, count*si4, data);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//registerGLvbo(posVbo[curPosRead]);
		registerGLBufferObject(posVbo[curPosRead], &m_cuda_posvbo_resource[curPosRead]);

	} else
		copyToDevice(dVel[curVelRead], data, start*si4, count*si4);
}
