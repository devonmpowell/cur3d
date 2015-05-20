
#include <stdio.h>
#include <cuda_runtime.h>

#include "cur3d.h"

// forward declarations
__global__ void cur3d_vox_kernel(cur3d_element* elems, r3d_int nelem, r3d_real* rho, r3d_dvec3 n, r3d_rvec3 d);
__host__ void cur3d_err(cudaError_t err, char* msg);

__device__ r3d_real cur3d_clip_and_reduce(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d);

__device__ void cur3du_cumsum(r3d_int* arr);
__device__ void cur3du_get_aabb(cur3d_element tet, r3d_dvec3 n, r3d_rvec3 d, r3d_dvec3 &vmin, r3d_dvec3 &vmax);
__device__ r3d_int cur3du_num_clip(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d);
__device__ void cur3du_init_box(r3d_poly* poly, r3d_rvec3 rbounds[2]);
__device__ r3d_real cur3du_orient(cur3d_element tet);
__device__ void cur3du_tet_faces_from_verts(r3d_rvec3* verts, r3d_plane* faces);




// useful macros
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
#define CLIP_MASK 0x80
#define dot(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define wav(va, wa, vb, wb, vr) {			\
	vr.x = (wa*va.x + wb*vb.x)/(wa + wb);	\
	vr.y = (wa*va.y + wb*vb.y)/(wa + wb);	\
	vr.z = (wa*va.z + wb*vb.z)/(wa + wb);	\
}
#define norm(v) {					\
	r3d_real tmplen = sqrt(dot(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}

// for re-indexing row-major voxel corners
__constant__ r3d_int cur3d_vv[8] = {0, 4, 3, 7, 1, 5, 2, 6};


__host__ void cur3d_voxelize_elements(cur3d_element* elems_h, r3d_int nelem, r3d_real* rho_h, r3d_dvec3 n, r3d_rvec3 d) {

	setbuf(stdout, NULL);

	cudaError_t e = cudaSuccess;

	// Allocate target grid on the device
	r3d_long ntot = n.i*n.j*n.k;
	r3d_real* rho_d;
	cudaMalloc((void**) &rho_d, ntot*sizeof(r3d_real));
	/*cudaMemset((void*) rho_d, 0, ntot*sizeof(r3d_real));*/
	cudaMemcpy(rho_d, rho_h, ntot*sizeof(r3d_real), cudaMemcpyHostToDevice);

	// Allocate and copy element buffer to the device
	cur3d_element* elems_d;
	cudaMalloc((void**) &elems_d, nelem*sizeof(cur3d_element));
	cudaMemcpy(elems_d, elems_h, nelem*sizeof(cur3d_element), cudaMemcpyHostToDevice);

	/*printf("Launching voxelization kernel, %d SMs * %d threads/SM = %d threads\n", NUM_SM, THREADS_PER_SM, NUM_SM*THREADS_PER_SM);*/
	cur3d_vox_kernel<<<NUM_SM, THREADS_PER_SM>>>(elems_d, nelem, rho_d, n, d);
		e = cudaGetLastError();
		cur3d_err(e, "kernel call");


	// TODO: this needs to be a reduction...
	cudaMemcpy(rho_h, rho_d, ntot*sizeof(r3d_real), cudaMemcpyDeviceToHost);

	// free device arrays
	cudaFree(rho_d);
	cudaFree(elems_d);

	return;
}

__global__ void cur3d_vox_kernel(cur3d_element* elems, r3d_int nelem, r3d_real* rho, r3d_dvec3 n, r3d_rvec3 d) {


	// voxel ring buffer
	// TODO: group clip-and-reduce operations by number of clip faces
	__shared__ r3d_int face_voxels[2*THREADS_PER_SM];
	__shared__ r3d_int face_tets[2*THREADS_PER_SM];
	__shared__ r3d_int vbuf_start, vbuf_end; 

	r3d_int tag_face;
	r3d_int nclip;

	__shared__ r3d_int cuminds[THREADS_PER_SM];

	// cumulative voxel offsets
	__shared__ r3d_int voxel_offsets[THREADS_PER_SM];
	__shared__ r3d_int voxels_per_block;

	// working vars
	cur3d_element tet;
	r3d_dvec3 vmin, vmax, vn, gidx; // voxel index range
	r3d_int vflat; // counters and such
	r3d_int tid; // local (shared memory) tet id
	r3d_int gid; // global tet id
	r3d_int vid; // local (shared memory) voxel id
	r3d_int btm;
	r3d_int top;
	
	// STEP 1
	// calculate offsets of each tet in the global voxel array
	// assumes that the total tet batch is <= GPU threads
	tid = threadIdx.x;
 	gid = blockIdx.x*blockDim.x + tid;
	voxel_offsets[tid] = 0;
	if(gid < nelem) {
		cur3du_get_aabb(elems[gid], n, d, vmin, vmax);
		voxel_offsets[tid] = (vmax.i - vmin.i)*(vmax.j - vmin.j)*(vmax.k - vmin.k); 
	}
	if(threadIdx.x == blockDim.x - 1)
		voxels_per_block = voxel_offsets[threadIdx.x];
	cur3du_cumsum(voxel_offsets);
	if(threadIdx.x == blockDim.x - 1)
		voxels_per_block += voxel_offsets[threadIdx.x];
	__syncthreads();

	// STEP 2
	// process all voxels in the AABBs and bin into separate buffers
	// for face and interior voxels
	// each thread gets one voxel
	if(threadIdx.x == 0) {
		vbuf_start = 0;
		vbuf_end = 0;
	}
	__syncthreads();
	for(vid = threadIdx.x; vid < voxels_per_block; vid += blockDim.x) {

		// binary search through cumulative voxel indices
		// to get the correct tet
		btm = 0;
		top = THREADS_PER_SM; 
		tid = (btm + top)/2;
		while(vid < voxel_offsets[tid] || vid >= voxel_offsets[tid+1]) {
			if(vid < voxel_offsets[tid]) top = tid;
			else btm = tid + 1;
			tid = (btm + top)/2;
		}
	 	gid = blockIdx.x*blockDim.x + tid;
		tet = elems[gid];

		// recompute the AABB for this tet	
		// to get the grid index of this voxel
		cur3du_get_aabb(tet, n, d, vmin, vmax);
		vn.i = vmax.i - vmin.i;
		vn.j = vmax.j - vmin.j;
		vn.k = vmax.k - vmin.k;
		vflat = vid - voxel_offsets[tid]; 
		gidx.i = vflat/(vn.j*vn.k); 
		gidx.j = (vflat - vn.j*vn.k*gidx.i)/vn.k;
		gidx.k = vflat - vn.j*vn.k*gidx.i - vn.k*gidx.j;
		gidx.i += vmin.i; gidx.j += vmin.j; gidx.k += vmin.k;

		// check the voxel against the tet faces
		nclip = cur3du_num_clip(tet, gidx, d);

		tag_face = 0;
		if(nclip == 0) // completely contained voxel 
			atomicAdd(&rho[n.j*n.k*gidx.i + n.k*gidx.j + gidx.k], tet.mass/(fabs(cur3du_orient(tet)) + 1.0e-99));
		else if(nclip > 0) // voxel must be clipped
			tag_face = 1;

		__syncthreads();

		// STEP 3
		// accumulate face voxels to a ring buffer
		// parallel scan to get indices, then parallel write to the ring buffer
		cuminds[threadIdx.x] = tag_face;
		cur3du_cumsum(cuminds);
		if(tag_face) {
			face_voxels[(vbuf_end + cuminds[threadIdx.x])%(2*THREADS_PER_SM)] = n.j*n.k*gidx.i + n.k*gidx.j + gidx.k;
			face_tets[(vbuf_end + cuminds[threadIdx.x])%(2*THREADS_PER_SM)] = tid;
		}
		if(threadIdx.x == blockDim.x - 1)
			vbuf_end += cuminds[threadIdx.x] + tag_face;
		__syncthreads();

		// STEP 4
		// parallel reduction of face voxels (1 per thread)
		if(vbuf_end - vbuf_start >= THREADS_PER_SM) {

			// recompute i, j, k, faces for this voxel
			vflat = face_voxels[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]; 
			gidx.i = vflat/(n.j*n.k); 
			gidx.j = (vflat - n.j*n.k*gidx.i)/n.k;
			gidx.k = vflat - n.j*n.k*gidx.i - n.k*gidx.j;
			tet = elems[blockIdx.x*blockDim.x + face_tets[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]];

			// clip and reduce to grid
			atomicAdd(&rho[vflat], cur3d_clip_and_reduce(tet, gidx, d));

			// shift ring buffer head
			if(threadIdx.x == 0)
				vbuf_start += THREADS_PER_SM;
		} 
		__syncthreads();
	}

	// STEP 5
	// clean up any face voxels remaining in the ring buffer
	if(threadIdx.x < vbuf_end - vbuf_start) {

		// recompute i, j, k, faces for this voxel
		vflat = face_voxels[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]; 
		gidx.i = vflat/(n.j*n.k); 
		gidx.j = (vflat - n.j*n.k*gidx.i)/n.k;
		gidx.k = vflat - n.j*n.k*gidx.i - n.k*gidx.j;
		tet = elems[blockIdx.x*blockDim.x + face_tets[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]];

		// clip and reduce to grid
		atomicAdd(&rho[vflat], cur3d_clip_and_reduce(tet, gidx, d));
	}
}

__device__ r3d_real cur3d_reduce(r3d_poly* poly) {

	// var declarations
	r3d_real locvol;
	unsigned char v, np, m;
	unsigned char vcur, vnext, pnext, vstart;
	r3d_rvec3 v0, v1, v2; 

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
	
	// for keeping track of which edges have been traversed
	unsigned char emarks[R3D_MAX_VERTS][3];
	memset((void*) &emarks, 0, sizeof(emarks));

	// stack for edges
	r3d_int nvstack;
	unsigned char vstack[2*R3D_MAX_VERTS];

	// find the first unclipped vertex
	vcur = R3D_MAX_VERTS;
	for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
		if(!(vertbuffer[v].orient.fflags & CLIP_MASK)) vcur = v;
	
	// return if all vertices have been clipped
	if(vcur == R3D_MAX_VERTS) return 0.0;

	locvol = 0;

	// stack implementation
	nvstack = 0;
	vstack[nvstack++] = vcur;
	vstack[nvstack++] = 0;

	while(nvstack > 0) {
		
		// get the next unmarked edge
		do {
			pnext = vstack[--nvstack];
			vcur = vstack[--nvstack];
		} while(emarks[vcur][pnext] && nvstack > 0);
		if(emarks[vcur][pnext] && nvstack == 0) break; 

		// initialize face looping
		emarks[vcur][pnext] = 1;
		vstart = vcur;
		v0 = vertbuffer[vstart].pos;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// move to the second edge
		for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
		vcur = vnext;
		pnext = (np+1)%3;
		emarks[vcur][pnext] = 1;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// make a triangle fan using edges
		// and first vertex
		while(vnext != vstart) {

			v2 = vertbuffer[vcur].pos;
			v1 = vertbuffer[vnext].pos;

			locvol += ONE_SIXTH*(-(v2.x*v1.y*v0.z) + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
				   	- v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z); 

			// move to the next edge
			for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
			vcur = vnext;
			pnext = (np+1)%3;
			emarks[vcur][pnext] = 1;
			vnext = vertbuffer[vcur].pnbrs[pnext];
			vstack[nvstack++] = vcur;
			vstack[nvstack++] = (pnext+1)%3;
		}
	}
	return locvol;
}


__device__ r3d_real cur3d_clip_and_reduce(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d) {

	r3d_real moments[10];
	r3d_poly poly;
	r3d_plane faces[4];
	r3d_real gor;
	r3d_rvec3 rbounds[2] = {
		{-0.5*d.x, -0.5*d.y, -0.5*d.z}, 
		{0.5*d.x, 0.5*d.y, 0.5*d.z} 
	};
	r3d_int v, f, ii, jj, kk;
	r3d_real tetvol;
	r3d_rvec3 gpt;
	unsigned char andcmp;

	tetvol = cur3du_orient(tet);
	if(tetvol < 0.0) {
		gpt = tet.pos[2];
		tet.pos[2] = tet.pos[3];
		tet.pos[3] = gpt;
		tetvol = -tetvol;
	}
	cur3du_tet_faces_from_verts(tet.pos, faces);

	// test the voxel against tet faces
	for(ii = 0; ii < 2; ++ii)
	for(jj = 0; jj < 2; ++jj)
	for(kk = 0; kk < 2; ++kk) {
		gpt.x = (ii + gidx.i)*d.x; gpt.y = (jj + gidx.j)*d.y; gpt.z = (kk + gidx.k)*d.z;
		v = cur3d_vv[4*ii + 2*jj + kk];
		poly.verts[v].orient.fflags = 0x00;
		for(f = 0; f < 4; ++f) {
			gor = faces[f].d + dot(gpt, faces[f].n);
			if(gor > 0.0) poly.verts[v].orient.fflags |= (1 << f);
			poly.verts[v].orient.fdist[f] = gor;
		}
	}

	andcmp = 0x0f;
	for(v = 0; v < 8; ++v) 
		andcmp &= poly.verts[v].orient.fflags;

	cur3du_init_box(&poly, rbounds);

	//// CLIP /////

	// variable declarations
	r3d_int nvstack;
	unsigned char vstack[4*R3D_MAX_VERTS];
	unsigned char ff, np, vcur, vprev, firstnewvert, prevnewvert;
	unsigned char fmask, ffmask;

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly.verts; 
	r3d_int* nverts = &poly.nverts; 
			
	for(f = 0; f < 4; ++f) {

		// go to the next active clip face
		fmask = (1 << f);
		while((andcmp & fmask) && f < 4)
			fmask = (1 << ++f);
		if(f == 4) break;

		// find the first vertex lying outside of the face
		// only need to find one (taking advantage of convexity)
		vcur = R3D_MAX_VERTS;
		for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
			if(!(vertbuffer[v].orient.fflags & (CLIP_MASK | fmask))) vcur = v;
		if(vcur == R3D_MAX_VERTS) continue; // TODO: can we do better here in terms of warp divergence?
		
		// push the first three edges and mark the starting vertex
		// as having been clipped
		nvstack = 0;
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = vertbuffer[vcur].pnbrs[1];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = vertbuffer[vcur].pnbrs[0];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = vertbuffer[vcur].pnbrs[2];
		vertbuffer[vcur].orient.fflags |= CLIP_MASK;
		firstnewvert = *nverts;
		prevnewvert = R3D_MAX_VERTS; 

		// traverse edges and clip
		// this is ordered very carefully to preserve edge connectivity
		while(nvstack > 0) {

			// get the next unclipped vertex
			do {
				vcur = vstack[--nvstack];
				vprev = vstack[--nvstack];
			} while((vertbuffer[vcur].orient.fflags & CLIP_MASK) && nvstack > 0);
			if((vertbuffer[vcur].orient.fflags & CLIP_MASK) && nvstack == 0) break; 

			// check whether this vertex is inside the face
			// if so, clip the edge and push the new vertex to vertbuffer
			if(vertbuffer[vcur].orient.fflags & fmask) {

				// compute the intersection point using a weighted
				// average of perpendicular distances to the plane
				wav(vertbuffer[vcur].pos, -vertbuffer[vprev].orient.fdist[f],
					vertbuffer[vprev].pos, vertbuffer[vcur].orient.fdist[f],
					vertbuffer[*nverts].pos);

				// doubly link to vcur
				for(np = 0; np < 3; ++np) if(vertbuffer[vcur].pnbrs[np] == vprev) break;
				vertbuffer[vcur].pnbrs[np] = *nverts;
				vertbuffer[*nverts].pnbrs[0] = vcur;

				// doubly link to previous new vert
				vertbuffer[*nverts].pnbrs[2] = prevnewvert; 
				vertbuffer[prevnewvert].pnbrs[1] = *nverts;

				// do face intersections and flags
				vertbuffer[*nverts].orient.fflags = 0x00;
				for(ff = f + 1; ff < 4; ++ff) {

					// TODO: might not need this one...
					/*ffmask = (1 << ff);*/
					/*while((andcmp & ffmask) && ff < 4)*/
						/*ffmask = (1 << ++ff);*/
					/*if(ff == 4) break;*/

					// skip if all verts are inside ff
					ffmask = (1 << ff); 
					if(andcmp & ffmask) continue;

					// weighted average keeps us in a relative coordinate system
					vertbuffer[*nverts].orient.fdist[ff] = 
							(vertbuffer[vprev].orient.fdist[ff]*vertbuffer[vcur].orient.fdist[f] 
							- vertbuffer[vprev].orient.fdist[f]*vertbuffer[vcur].orient.fdist[ff])
							/(vertbuffer[vcur].orient.fdist[f] - vertbuffer[vprev].orient.fdist[f]);
					if(vertbuffer[*nverts].orient.fdist[ff] > 0.0) vertbuffer[*nverts].orient.fflags |= ffmask;
				}

				prevnewvert = (*nverts)++;
			}
			else {

				// otherwise, determine the left and right vertices
				// (ordering is important) and push to the traversal stack
				for(np = 0; np < 3; ++np) if(vertbuffer[vcur].pnbrs[np] == vprev) break;

				// mark the vertex as having been clipped
				vertbuffer[vcur].orient.fflags |= CLIP_MASK;

				// push the next verts to the stack
				vstack[nvstack++] = vcur;
				vstack[nvstack++] = vertbuffer[vcur].pnbrs[(np+2)%3];
				vstack[nvstack++] = vcur;
				vstack[nvstack++] = vertbuffer[vcur].pnbrs[(np+1)%3];
			}
		}

		// close the clipped face
		vertbuffer[firstnewvert].pnbrs[2] = *nverts-1;
		vertbuffer[prevnewvert].pnbrs[1] = firstnewvert;
	}

	////// REDUCE ///////

#if 0
	// var declarations
	r3d_real locvol;
	unsigned char m;
	unsigned char vnext, pnext, vstart;
	r3d_rvec3 v0, v1, v2; 

	r3d_int polyorder = 0;
	
	// for keeping track of which edges have been traversed
	unsigned char emarks[R3D_MAX_VERTS][3];
	memset((void*) &emarks, 0, sizeof(emarks));

	// zero the moments
	for(m = 0; m < 10; ++m)
		moments[m] = 0.0;

	// find the first unclipped vertex
	vcur = R3D_MAX_VERTS;
	for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
		if(!(vertbuffer[v].orient.fflags & CLIP_MASK)) vcur = v;
	
	// return if all vertices have been clipped
	if(vcur == R3D_MAX_VERTS) return 0.0;

	// stack implementation
	nvstack = 0;
	vstack[nvstack++] = vcur;
	vstack[nvstack++] = 0;

	while(nvstack > 0) {
		
		pnext = vstack[--nvstack];
		vcur = vstack[--nvstack];

		// skip this edge if we have marked it
		if(emarks[vcur][pnext]) continue;


		// initialize face looping
		emarks[vcur][pnext] = 1;
		vstart = vcur;
		v0 = vertbuffer[vstart].pos;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// move to the second edge
		for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
		vcur = vnext;
		pnext = (np+1)%3;
		emarks[vcur][pnext] = 1;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// make a triangle fan using edges
		// and first vertex
		while(vnext != vstart) {

			v2 = vertbuffer[vcur].pos;
			v1 = vertbuffer[vnext].pos;

			locvol = ONE_SIXTH*(-(v2.x*v1.y*v0.z) + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
				   	- v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z); 

			moments[0] += locvol; 
			if(polyorder >= 1) {
				moments[1] += locvol*0.25*(v0.x + v1.x + v2.x);
				moments[2] += locvol*0.25*(v0.y + v1.y + v2.y);
				moments[3] += locvol*0.25*(v0.z + v1.z + v2.z);
			}
			if(polyorder >= 2) {
				moments[4] += locvol*0.1*(v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + v1.x*v2.x + v0.x*(v1.x + v2.x));
				moments[5] += locvol*0.1*(v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + v1.y*v2.y + v0.y*(v1.y + v2.y));
				moments[6] += locvol*0.1*(v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + v1.z*v2.z + v0.z*(v1.z + v2.z));
				moments[7] += locvol*0.05*(v2.x*v0.y + v2.x*v1.y + 2*v2.x*v2.y + v0.x*(2*v0.y + v1.y + v2.y) + v1.x*(v0.y + 2*v1.y + v2.y));
				moments[8] += locvol*0.05*(v2.y*v0.z + v2.y*v1.z + 2*v2.y*v2.z + v0.y*(2*v0.z + v1.z + v2.z) + v1.y*(v0.z + 2*v1.z + v2.z));
				moments[9] += locvol*0.05*(v2.x*v0.z + v2.x*v1.z + 2*v2.x*v2.z + v0.x*(2*v0.z + v1.z + v2.z) + v1.x*(v0.z + 2*v1.z + v2.z));
			}

			// move to the next edge
			for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
			vcur = vnext;
			pnext = (np+1)%3;
			emarks[vcur][pnext] = 1;
			vnext = vertbuffer[vcur].pnbrs[pnext];
			vstack[nvstack++] = vcur;
			vstack[nvstack++] = (pnext+1)%3;
		}
	}
#endif





	return tet.mass/(tetvol + 1.0e-99)*cur3d_reduce(&poly)/(d.x*d.y*d.z);

}




// parallel prefix scan in shared memory
// scan is in-place, so the result replaces the input array
// assumes input of length THREADS_PER_SM
// from GPU Gems 3, ch. 39
__device__ void cur3du_cumsum(r3d_int* arr) {

	// TODO: faster scan operation might be needed
	// (i.e. naive but less memory-efficient)
	r3d_int offset, d, ai, bi, t;

	// build the sum in place up the tree
	offset = 1;
	for (d = THREADS_PER_SM>>1; d > 0; d >>= 1) {
		__syncthreads();
		if (threadIdx.x < d) {
			ai = offset*(2*threadIdx.x+1)-1;
			bi = offset*(2*threadIdx.x+2)-1;
			arr[bi] += arr[ai];
		}
		offset *= 2;
	}

	// clear the last element
	if (threadIdx.x == 0)
		arr[THREADS_PER_SM - 1] = 0;   

	// traverse down the tree building the scan in place
	for (d = 1; d < THREADS_PER_SM; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (threadIdx.x < d) {
			ai = offset*(2*threadIdx.x+1)-1;
			bi = offset*(2*threadIdx.x+2)-1;
			t = arr[ai];
			arr[ai] = arr[bi];
			arr[bi] += t;
		}
	}
	__syncthreads();
}

__device__ void cur3du_get_aabb(cur3d_element tet, r3d_dvec3 n, r3d_rvec3 d, r3d_dvec3 &vmin, r3d_dvec3 &vmax) {

		// get the AABB for this tet
		// and clamp to destination grid dims
		r3d_int v;
		r3d_rvec3 rmin, rmax;
		rmin.x = 1.0e10; rmin.y = 1.0e10; rmin.z = 1.0e10;
		rmax.x = -1.0e10; rmax.y = -1.0e10; rmax.z = -1.0e10;
		for(v = 0; v < 4; ++v) {
			if(tet.pos[v].x < rmin.x) rmin.x = tet.pos[v].x;
			if(tet.pos[v].x > rmax.x) rmax.x = tet.pos[v].x;
			if(tet.pos[v].y < rmin.y) rmin.y = tet.pos[v].y;
			if(tet.pos[v].y > rmax.y) rmax.y = tet.pos[v].y;
			if(tet.pos[v].z < rmin.z) rmin.z = tet.pos[v].z;
			if(tet.pos[v].z > rmax.z) rmax.z = tet.pos[v].z;
		}
		vmin.i = floor(rmin.x/d.x);
		vmin.j = floor(rmin.y/d.y);
		vmin.k = floor(rmin.z/d.z);
		vmax.i = ceil(rmax.x/d.x);
		vmax.j = ceil(rmax.y/d.y);
		vmax.k = ceil(rmax.z/d.z);
		if(vmin.i < 0) vmin.i = 0;
		if(vmin.j < 0) vmin.j = 0;
		if(vmin.k < 0) vmin.k = 0;
		if(vmax.i > n.i) vmax.i = n.i;
		if(vmax.j > n.j) vmax.j = n.j;
		if(vmax.k > n.k) vmax.k = n.k;

}

__device__ r3d_int cur3du_num_clip(cur3d_element tet, r3d_dvec3 gidx, r3d_rvec3 d) {

	r3d_real tetvol;
	r3d_plane faces[4];
	r3d_rvec3 gpt;
	r3d_int f, ii, jj, kk;
	unsigned char andcmp, orcmp, fflags;
	/*r3d_int nclip;*/

	// properly orient the tet
	tetvol = cur3du_orient(tet);
	if(tetvol < 0.0) {
		gpt = tet.pos[2];
		tet.pos[2] = tet.pos[3];
		tet.pos[3] = gpt;
		tetvol = -tetvol;
	}

	// TODO: This does some sqrts that might not be needed...
	cur3du_tet_faces_from_verts(tet.pos, faces);
	
	// test the bin corners against tet faces to determine voxel type
	orcmp = 0x00;
	andcmp = 0x0f;
	for(ii = 0; ii < 2; ++ii)
	for(jj = 0; jj < 2; ++jj)
	for(kk = 0; kk < 2; ++kk) {
		gpt.x = (ii + gidx.i)*d.x; gpt.y = (jj + gidx.j)*d.y; gpt.z = (kk + gidx.k)*d.z;
		fflags = 0x00;
		for(f = 0; f < 4; ++f) 
			if(faces[f].d + dot(gpt, faces[f].n) > 0.0) fflags |= (1 << f);
		andcmp &= fflags;
		orcmp |= fflags;
	}

	// if the voxel is completely outside the tet, return -1
	if(orcmp < 0x0f) return -1;
	
	// else, return the number of faces to be clipped against
	return 4 - __popc(andcmp);
}

__device__ void cur3du_init_box(r3d_poly* poly, r3d_rvec3 rbounds[2]) {

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
	
	*nverts = 8;
	vertbuffer[0].pnbrs[0] = 1;	
	vertbuffer[0].pnbrs[1] = 4;	
	vertbuffer[0].pnbrs[2] = 3;	
	vertbuffer[1].pnbrs[0] = 2;	
	vertbuffer[1].pnbrs[1] = 5;	
	vertbuffer[1].pnbrs[2] = 0;	
	vertbuffer[2].pnbrs[0] = 3;	
	vertbuffer[2].pnbrs[1] = 6;	
	vertbuffer[2].pnbrs[2] = 1;	
	vertbuffer[3].pnbrs[0] = 0;	
	vertbuffer[3].pnbrs[1] = 7;	
	vertbuffer[3].pnbrs[2] = 2;	
	vertbuffer[4].pnbrs[0] = 7;	
	vertbuffer[4].pnbrs[1] = 0;	
	vertbuffer[4].pnbrs[2] = 5;	
	vertbuffer[5].pnbrs[0] = 4;	
	vertbuffer[5].pnbrs[1] = 1;	
	vertbuffer[5].pnbrs[2] = 6;	
	vertbuffer[6].pnbrs[0] = 5;	
	vertbuffer[6].pnbrs[1] = 2;	
	vertbuffer[6].pnbrs[2] = 7;	
	vertbuffer[7].pnbrs[0] = 6;	
	vertbuffer[7].pnbrs[1] = 3;	
	vertbuffer[7].pnbrs[2] = 4;	
	vertbuffer[0].pos.x = rbounds[0].x; 
	vertbuffer[0].pos.y = rbounds[0].y; 
	vertbuffer[0].pos.z = rbounds[0].z; 
	vertbuffer[1].pos.x = rbounds[1].x; 
	vertbuffer[1].pos.y = rbounds[0].y; 
	vertbuffer[1].pos.z = rbounds[0].z; 
	vertbuffer[2].pos.x = rbounds[1].x; 
	vertbuffer[2].pos.y = rbounds[1].y; 
	vertbuffer[2].pos.z = rbounds[0].z; 
	vertbuffer[3].pos.x = rbounds[0].x; 
	vertbuffer[3].pos.y = rbounds[1].y; 
	vertbuffer[3].pos.z = rbounds[0].z; 
	vertbuffer[4].pos.x = rbounds[0].x; 
	vertbuffer[4].pos.y = rbounds[0].y; 
	vertbuffer[4].pos.z = rbounds[1].z; 
	vertbuffer[5].pos.x = rbounds[1].x; 
	vertbuffer[5].pos.y = rbounds[0].y; 
	vertbuffer[5].pos.z = rbounds[1].z; 
	vertbuffer[6].pos.x = rbounds[1].x; 
	vertbuffer[6].pos.y = rbounds[1].y; 
	vertbuffer[6].pos.z = rbounds[1].z; 
	vertbuffer[7].pos.x = rbounds[0].x; 
	vertbuffer[7].pos.y = rbounds[1].y; 
	vertbuffer[7].pos.z = rbounds[1].z; 
}

__device__ r3d_real cur3du_orient(cur3d_element tet) {
	r3d_real adx, bdx, cdx;
	r3d_real ady, bdy, cdy;
	r3d_real adz, bdz, cdz;
	adx = tet.pos[0].x - tet.pos[3].x;
	bdx = tet.pos[1].x - tet.pos[3].x;
	cdx = tet.pos[2].x - tet.pos[3].x;
	ady = tet.pos[0].y - tet.pos[3].y;
	bdy = tet.pos[1].y - tet.pos[3].y;
	cdy = tet.pos[2].y - tet.pos[3].y;
	adz = tet.pos[0].z - tet.pos[3].z;
	bdz = tet.pos[1].z - tet.pos[3].z;
	cdz = tet.pos[2].z - tet.pos[3].z;
	return -ONE_SIXTH*(adx * (bdy * cdz - bdz * cdy)
			+ bdx * (cdy * adz - cdz * ady)
			+ cdx * (ady * bdz - adz * bdy));
}

  
__host__ void cur3d_err(cudaError_t err, char* msg) {
	if (err != cudaSuccess) {
		printf("CUDA Error: %s at %s.\n", cudaGetErrorString(err), msg);
		exit(0);
	}
}

__device__ void cur3du_tet_faces_from_verts(r3d_rvec3* verts, r3d_plane* faces) {
	// compute unit face normals and distances to origin
	r3d_rvec3 tmpcent;
	faces[0].n.x = ((verts[3].y - verts[1].y)*(verts[2].z - verts[1].z) 
			- (verts[2].y - verts[1].y)*(verts[3].z - verts[1].z));
	faces[0].n.y = ((verts[2].x - verts[1].x)*(verts[3].z - verts[1].z) 
			- (verts[3].x - verts[1].x)*(verts[2].z - verts[1].z));
	faces[0].n.z = ((verts[3].x - verts[1].x)*(verts[2].y - verts[1].y) 
			- (verts[2].x - verts[1].x)*(verts[3].y - verts[1].y));
	norm(faces[0].n);
	tmpcent.x = ONE_THIRD*(verts[1].x + verts[2].x + verts[3].x);
	tmpcent.y = ONE_THIRD*(verts[1].y + verts[2].y + verts[3].y);
	tmpcent.z = ONE_THIRD*(verts[1].z + verts[2].z + verts[3].z);
	faces[0].d = -dot(faces[0].n, tmpcent);

	faces[1].n.x = ((verts[2].y - verts[0].y)*(verts[3].z - verts[2].z) 
			- (verts[2].y - verts[3].y)*(verts[0].z - verts[2].z));
	faces[1].n.y = ((verts[3].x - verts[2].x)*(verts[2].z - verts[0].z) 
			- (verts[0].x - verts[2].x)*(verts[2].z - verts[3].z));
	faces[1].n.z = ((verts[2].x - verts[0].x)*(verts[3].y - verts[2].y) 
			- (verts[2].x - verts[3].x)*(verts[0].y - verts[2].y));
	norm(faces[1].n);
	tmpcent.x = ONE_THIRD*(verts[2].x + verts[3].x + verts[0].x);
	tmpcent.y = ONE_THIRD*(verts[2].y + verts[3].y + verts[0].y);
	tmpcent.z = ONE_THIRD*(verts[2].z + verts[3].z + verts[0].z);
	faces[1].d = -dot(faces[1].n, tmpcent);

	faces[2].n.x = ((verts[1].y - verts[3].y)*(verts[0].z - verts[3].z) 
			- (verts[0].y - verts[3].y)*(verts[1].z - verts[3].z));
	faces[2].n.y = ((verts[0].x - verts[3].x)*(verts[1].z - verts[3].z) 
			- (verts[1].x - verts[3].x)*(verts[0].z - verts[3].z));
	faces[2].n.z = ((verts[1].x - verts[3].x)*(verts[0].y - verts[3].y) 
			- (verts[0].x - verts[3].x)*(verts[1].y - verts[3].y));
	norm(faces[2].n);
	tmpcent.x = ONE_THIRD*(verts[3].x + verts[0].x + verts[1].x);
	tmpcent.y = ONE_THIRD*(verts[3].y + verts[0].y + verts[1].y);
	tmpcent.z = ONE_THIRD*(verts[3].z + verts[0].z + verts[1].z);
	faces[2].d = -dot(faces[2].n, tmpcent);

	faces[3].n.x = ((verts[0].y - verts[2].y)*(verts[1].z - verts[0].z) 
			- (verts[0].y - verts[1].y)*(verts[2].z - verts[0].z));
	faces[3].n.y = ((verts[1].x - verts[0].x)*(verts[0].z - verts[2].z) 
			- (verts[2].x - verts[0].x)*(verts[0].z - verts[1].z));
	faces[3].n.z = ((verts[0].x - verts[2].x)*(verts[1].y - verts[0].y) 
			- (verts[0].x - verts[1].x)*(verts[2].y - verts[0].y));
	norm(faces[3].n);
	tmpcent.x = ONE_THIRD*(verts[0].x + verts[1].x + verts[2].x);
	tmpcent.y = ONE_THIRD*(verts[0].y + verts[1].y + verts[2].y);
	tmpcent.z = ONE_THIRD*(verts[0].z + verts[1].z + verts[2].z);
	faces[3].d = -dot(faces[3].n, tmpcent);
}

