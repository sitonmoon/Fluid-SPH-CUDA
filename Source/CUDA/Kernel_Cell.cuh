
//  Grid, Sort
//----------------------------------------------------------------------------------------------------------------------------

__device__ int3 calcGridPos(float3 p)	//  calculate position in uniform grid
{
	int3 gridPos;
	float3 gp = (p- par.worldMin) / par.cellSize;
	gridPos.x = (int)floor(gp.x);	//gridPos.x = max(0, min(gridPos.x, par.gridSize.x-1));	// not needed
	gridPos.y = (int)floor(gp.y);	//gridPos.y = max(0, min(gridPos.y, par.gridSize.y-1));  //(clamping to edges)
	gridPos.z = (int)floor(gp.z);	//gridPos.z = max(0, min(gridPos.z, par.gridSize.z-1));
	return gridPos;
}

__device__ uint calcGridHash(int3 gridPos)	//  calculate address in grid from position (clamping to edges)
{
	//gridPos.x = gridPos.x & (par.gridSize.x-1);  // wrap grid, assumes size is power of 2
	//gridPos.y = gridPos.y & (par.gridSize.y-1);
	//gridPos.z = gridPos.z & (par.gridSize.z-1);
	//return __mul24(gridPos.z, __mul24(gridPos.y,gridPos.x)) + __mul24(gridPos.y,gridPos.x) + gridPos.x;
	return __umul24(__umul24(gridPos.z, par.gridSize.y), par.gridSize.x) + __umul24(gridPos.y, par.gridSize.x) + gridPos.x;
}

//  calculate grid hash value for each particle
__global__ void calcHashD(float4* pos, 
						  uint* particleHash, // output
						  uint* particleIndex, // output
						  uint  numParticles
						  )
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x; 
	if (index >= numParticles) return;

	volatile float4 p = pos[index];

	// get address in grid
	int3 gridPos  = calcGridPos(make_float3(p.x, p.y, p.z));
	uint gridHash = calcGridHash(gridPos);

	// store grid hash and particle index
	particleHash[index] = gridHash;
	particleIndex[index] = index;

}


/// rearrange particle data into sorted order,
/// and find the start of each cell in the sorted hash array

__global__ void reorderD(uint* particleHash, uint* particleIndex, uint* cellStart,uint* cellEnd,
						 float4* oldPos, float4* oldVel, float4* sortedPos, float4* sortedVel,uint numParticles)
{
	__shared__ uint sharedHash[257]; // blockSize + 1 elements
	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	uint hash = particleHash[index];

	// Load hash data into shared memory so that we can look 
	// at neighboring particle's hash value without loading
	// two hash values per thread
	sharedHash[threadIdx.x+1] = hash;

	if (index > 0 && threadIdx.x == 0)
	{
		// first thread in block must load neighbor particle hash
		sharedHash[0] = particleHash[index-1];
	}
	__syncthreads();

	// If this particle has a different cell index to the previous particle
	// then it must be the first particle in the cell,
	// so store the index of this particle in the cell.
	// As it isn't the first particle, it must also be the cell end of
	// the previous particle's cell

	if (index == 0 || hash != sharedHash[threadIdx.x])
	{
		cellStart[hash] = index;

		if (index > 0)
			cellEnd[sharedHash[threadIdx.x]] = index;
	}
	if (index == numParticles - 1)
	{
		cellEnd[hash] = index + 1;
	}
	// Now use the sorted index to reorder the pos and vel data
	uint sortedIndex = particleIndex[index];
	float4 pos = FETCH(oldPos, sortedIndex);  sortedPos[index] = pos;
	float4 vel = FETCH(oldVel, sortedIndex);  sortedVel[index] = vel;

}



//----------------------------------------------------------------------------------------------------------------------------
///  Collide
//----------------------------------------------------------------------------------------------------------------------------

//  collide two spheres using DEM method
__device__ float3 collideSpheres(float4 posAB, float4 velAB, float radiusAB)
{
	float3 relPos = make_float3(posAB);		// relative position
	float  dist = length(relPos);

	float3 force = make_float3(0.0f);
	if (dist < radiusAB)
	{
		float3 relVel = make_float3(velAB);		// relative velocity

		float3 norm = relPos / dist;		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);

		force = par.spring * (dist - radiusAB) * norm;		// spring
		force += par.damping * relVel;		//  damping
		force += par.shear * tanVel;		// tangential shear force
	}
	return force;
}

//  collide two spheres using DEM method
__device__ float3 collideSpheresR(float3 posAB, float3 relVel, float radiusAB)
{
	float  dist = length(posAB);
	float3 force = make_float3(0.0f);
	if (dist < radiusAB)
	{
		float3 norm = posAB / dist;		// relative tangential velocity
		//float3 tanVel = relVel - (dot(relVel, norm) * norm);

		force = par.spring * (dist - radiusAB) * norm;		// spring
		force += par.damping * relVel;		//  damping
		//force += par.shear * tanVel;		// tangential shear force
	}
	return force;
}


///----------------------------------------------------------------------------------------------------------------------------
///  Density
///----------------------------------------------------------------------------------------------------------------------------

///  density in Cell
__device__ float compDensCell(int3 gridPos, uint index,  float4 pos, float4* oldPos,  uint* particleHash, uint* cellStart)
{
	float dens = 0.0f;

	uint gridHash = calcGridHash(gridPos);
	uint bucketStart = FETCH(cellStart, gridHash);
	if (bucketStart == 0xffffffff)	return dens;

	//  iterate over particles in this cell
	uint endIndex = FETCH(cellEnd, gridHash);
	for (uint i=bucketStart; i < endIndex; i++)
	{
		uint cellData = particleHash[i];
		if (cellData != gridHash)  break;
		if (i != index)	// check not colliding with self
		{
			float4 pos2 = FETCH(oldPos, i);

			///  pair density
			float4 p = pos - pos2;  // relPos
			float r2 = p.x*p.x + p.y*p.y + p.z*p.z;

			if (r2 < par.h2)
			{
				float c = par.h2 - r2;	// W6(r,h)
				dens += pow(c, 3);
			}
		}
		else
		{
			dens += pow(par.h2, 3.f);  //self
		}
	}
	return dens;
}

///  compute Density  ------------------------------------------

__global__ void computeDensityD( float4* oldPos,  float* pressure, float* density,
								uint* particleHash, uint* cellStart, uint* cellEnd)
{
	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

	float4 pos = FETCH(oldPos, index);
	int3 gridPos = calcGridPos(make_float3(pos.x, pos.y, pos.z));

	float sum = 0.0f;

	for(int z=-1; z<=1; z++)
		for(int y=-1; y<=1; y++)
			for(int x=-1; x<=1; x++)
			{
				sum += compDensCell(gridPos + make_int3(x,y,z), index, pos, oldPos, particleHash, cellStart);
			}

	float dens = sum * par.Poly6Kern * par.particleMass;
	float pres = (dens - par.restDensity) * par.stiffness;
	//float pres = (pow(dens / par.restDensity, 7) - 1) * par.stiffness; //这里使用了Tait方程
	pressure[index] = pres;
	density[index] = dens;
}



///----------------------------------------------------------------------------------------------------------------------------
//   Force
///----------------------------------------------------------------------------------------------------------------------------

///  compute force in Cell

__device__ float3 compForceCell(int3 gridPos, uint index,
								float4& pos, float4& vel, float4* oldPos, float4* oldVel,
								float pres, float dens, float* pressure, float* density,
								uint* particleHash, uint* cellStart, uint* cellEnd)
{
	float3 force = make_float3(0.0f);
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = FETCH(cellStart, gridHash);
	if (startIndex == 0xffffffff)	return force;

	uint endIndex = FETCH(cellEnd, gridHash);
	//  iterate over particles in this cell
	for (uint i= startIndex; i < endIndex; i++)
	{
		uint cellData = particleHash[i];
		if (cellData != gridHash)  break;

		if (i != index)
		{
			float4 pos2 = FETCH(oldPos, i);	float4 vel2 = FETCH(oldVel, i);
			float pres2 = FETCH(pressure, i);	float dens2 = FETCH(density, i);

			//  SPH force for pair of particles
			float3 RelPos = make_float3(pos - pos2);
			float3 relVel = make_float3(vel2 - vel);
			float d12 = 1.0f / (dens * dens2);
			float p12 = pres + pres2;

			//  SPH force for pair of particles
			float3 fcur = make_float3(0.0f);
			float r = max(par.minDist, length(RelPos));	
			if (r < par.h)
			{	
				float h_r = par.h - r;
				float pterm = h_r * par.SpikyKern * p12/r;
				float vterm = par.LapKern * par.viscosity;

				fcur = pterm * RelPos/*dP*/ + vterm * relVel/*dV*/ ;
				fcur *= h_r*d12;
			}
			force += fcur;
		}
	}
	return force;
}


//  collide two particls for test
__device__ float3 collideParticls(float4& posAB, float4& velAB, float radiusAB)
{
	float3 relPos = make_float3(posAB);		// relative position
	float  dist = length(relPos);
	float3 force = make_float3(0.0f);
	if (dist < radiusAB)
	{
		float3 relVel = make_float3(velAB);		// relative velocity

		float3 norm = relPos / dist;		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);
		tanVel*=0.05;
		force = par.spring * (radiusAB- dist) * norm;		// spring
		force += -par.damping * relVel;		//  damping
		force += par.shear * tanVel;		// tangential shear force
	}
	return force;
}

// test simple collide
// collide a particle against all other particles in a given cell
__device__
	float3 collideCell(int3    gridPos,
	uint    index,
	float4  pos,
	float4  vel,
	float4 *oldPos,
	float4 *oldVel,
	uint	*particleHash,
	uint   *cellStart,
	uint   *cellEnd)
{
	uint gridHash = calcGridHash(gridPos);

	//get start of bucket for this cell
	uint startIndex = FETCH(cellStart, gridHash);

	float3 force = make_float3(0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = FETCH(cellEnd, gridHash);

		for (uint j=startIndex; j<endIndex; j++)
		{
			uint cellData = particleHash[j];
			if (cellData != gridHash)  break;

			if (j != index)                // check not colliding with self
			{
				float4 pos2 = FETCH(oldPos, j);

				// collide two particles
				force += collideParticls(pos-pos2, vel, par.particleR*2);
			}
		}
	}

	return force;
}
