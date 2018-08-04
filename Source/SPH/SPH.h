#pragma once

#include "..\CUDA\Params.cuh"
#include "..\pch\timer.h"
#include "..\SPH\Scene.h"



class cSPH
{
public:		///  Methods

	cSPH();  ~cSPH();
	void _InitMem(), _FreeMem();	bool bInitialized;

	void Update();	// simulate
	void dumpGrid();
	void Reset(int type), Drop(bool bRandom);  // init volume
	float3 DropPos;

	//  Scenes
	static void LoadOptions();
	void LoadScenes(), InitScene(), NextScene(bool chapter=false),PrevScene(bool chapter=false),UpdScene();
	vector<Scene> scenes;	Scene scn;	int curScene;

	//  utility
	float* getArray(bool pos);
	void setArray(bool pos, const float4* data, int start, int count);

	uint getPosBuffer() const 
	{ 
		return posVbo[curPosRead];  
	}

	uint createVBO(uint size);
	void colorRamp(float t, float *r);


public:		///  Data

	//  h-host CPU,  d-device GPU
	float4*  hPos,*hVel; 
	float4*  dPos[2], *dVel[2], *dSortedPos,*dSortedVel;
	uint  *m_hCellStart;
	uint*  hParHash,*dParHash, *dParIndex, *hCellStart,*dCellStart ,*dCellEnd;
	int*  hCounters,*dCounters[2];  // debug

	float* dPressure, *dDensity, *dDyeColor;

	uint posVbo[2], colorVbo;	//  GL vbo
	uint curPosRead, curVelRead, curPosWrite, curVelWrite;  // swap

	 float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
     float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

     struct cudaGraphicsResource *m_cuda_posvbo_resource[2]; // handles OpenGL-CUDA exchange
     struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

	//  timer-
	Timer tim;	//float t1,t2,t3,t4,t5;
};
