#include "header.h"

#include "..\SPH\SPH.h"
#include "..\CUDA\System.cuh"
#include "..\CUDA\radixsort.cuh"
#include "..\Graphics\param.h"
#include "..\App\App.h"
#include <fstream>

// dump grid information
void cSPH::dumpGrid()
{  
	SimParams& p = scn.params;
	copyFromDevice(m_hCellStart, dCellStart, 0, sizeof(uint)*p.numCells);
    uint maxCellSize = 0;
	ofstream dumptxt("dempGrid.txt");
	dumptxt<<"numCells "<< p.numCells<< endl;
	dumptxt<<"cellSize "<< p.cellSize.x <<endl;
    for (uint i=0; i<p.numCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
			dumptxt<<"Cell:"<<i<<" GridHash:"<<m_hCellStart[i] <<endl;

        }
    }
	printf("See dempGrid.txt \n");
	dumptxt.close();
}
///----------------------------------------------------------------------------------------------------------------------------------
///  Update
///----------------------------------------------------------------------------------------------------------------------------------
void cSPH::Update()
{
	if (!bInitialized)  return;
	/*-*/tim.update(true);

	//  update sim constants, when changed
	if (ParamBase::bChangedAny == true)
	{	ParamBase::bChangedAny = false;  /**/if (App::nChg < 20)  App::nChg += 10;
		App::updTabVis();/*-*/	App::updBrCnt();
		/*tim.update();	t5 = 1000.0*tim.dt;	/**/	}
	
	dPos[curPosRead] = (float4 *) mapGLBufferObject(&m_cuda_posvbo_resource[curPosRead]);
	dPos[curPosWrite] = (float4 *) mapGLBufferObject(&m_cuda_posvbo_resource[curPosWrite]);
	
	SimParams& p = scn.params;
	setParameters(&p);

	///  integrate  (and boundary)
	//hCounters[0] = 0;  hCounters[1] = 0;
	//**/copyToDevice(dCounters[curPosWrite], hCounters, 0, 4*sizeof(int));

	integrate(dPos[curPosRead], dVel[curVelRead], dPos[curPosWrite], dVel[curVelWrite],  p.numParticles/*, dCounters[curPosWrite]*/);

	swap(curPosRead, curPosWrite);
	//swap(curVelRead, curVelWrite);
	
	//// debug -slow
	////copyFromDevice(hPos, 0, posVbo[curPosRead], sizeof(float)*4*50/*p.numParticles*/);
	////copyFromDevice(hVel, 0, colorVbo2/*dVel[curVelRead], 0,*/, sizeof(float)*4*50/*p.numParticles*/);

	calcHash(dPos[curPosRead], dParHash, dParIndex, p.numParticles);

	//// sort particles based on hash
	sortParticles(dParHash, dParIndex, p.numParticles);
	//RadixSort((KeyValuePair*) dParHash[0], (KeyValuePair*) dParHash[1], p.numParticles,/*bits*/p.numCells >= 65536 ? 32 : 16);

	reorder(dPos[curPosRead], dVel[curVelWrite], dSortedPos, dSortedVel, dParHash, dParIndex, dCellStart, dCellEnd, p.numParticles, p.numCells);

	float4* clr2 = (float4 *) mapGLBufferObject(&m_cuda_colorvbo_resource);
	collide(dVel[curVelRead], clr2,
		dSortedPos, dSortedVel,  
		dPressure, dDensity, dDyeColor,//
		dParHash, dParIndex, dCellStart, dCellEnd, p.numParticles, p.numCells);
	unmapGLBufferObject(m_cuda_colorvbo_resource);
	
	unmapGLBufferObject(m_cuda_posvbo_resource[curPosRead]);
	unmapGLBufferObject(m_cuda_posvbo_resource[curPosWrite]);

	//swap(curVelRead, curVelWrite);
	
}
///----------------------------------------------------------------------------------------------------------------------------------
