
#include <stdio.h>
#include <cuda_runtime.h>

#include "cur3d.h"

// CUDA-specific forward declarations

__global__ void k_vox(cur3d_element* elems, r3d_int nelem, r3d_real* rho, r3d_dvec3 n, r3d_rvec3 d);

__device__ void cur3du_get_aabb(cur3d_element tet, r3d_dvec3 n, r3d_rvec3 d, r3d_dvec3 &vmin,
r3d_dvec3 &vmax);

__device__ r3d_real cur3d_clip_and_reduce(cur3d_element tet, r3d_int i, r3d_int j, r3d_int k,
r3d_rvec3 d);
__device__ void cur3d_clip_tet(r3d_poly* poly, unsigned char andcmp);
__device__ void cur3d_reduce(r3d_poly* poly, r3d_int polyorder, r3d_real* moments);
__device__ void cur3du_init_box(r3d_poly* poly, r3d_rvec3 rbounds[2]);
__device__ r3d_real cur3du_orient(r3d_rvec3 pa, r3d_rvec3 pb, r3d_rvec3 pc, r3d_rvec3 pd);
__device__ void cur3du_tet_faces_from_verts(r3d_rvec3* verts, r3d_plane* faces);
__host__ void cur3d_err(cudaError_t err, char* msg);



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
	k_vox<<<NUM_SM, THREADS_PER_SM>>>(elems_d, nelem, rho_d, n, d);
		e = cudaGetLastError();
		cur3d_err(e, "kernel call");


	// TODO: this needs to be a reduction...
	cudaMemcpy(rho_h, rho_d, ntot*sizeof(r3d_real), cudaMemcpyDeviceToHost);

	// free device arrays
	cudaFree(rho_d);
	cudaFree(elems_d);

	return;
}

// single-voxel kernel
__global__ void k_vox(cur3d_element* elems, r3d_int nelem, r3d_real* rho, r3d_dvec3 n, r3d_rvec3 d) {

	// voxels per tet
	__shared__ r3d_int numvox[THREADS_PER_SM]; // voxels per tet

	// starting indices for each warp (for scan operations)
	// TODO: factor of 2 needed for naive scan impementation
	// TODO: more memory-efficient scan implementation
	__shared__ r3d_int cuminds[2*THREADS_PER_SM]; 
	r3d_int pout, pin, offset; 

	// voxel ring buffer
	// TODO: use smaller ring buffer?
	// TODO: group clip-and-reduce operations by number of clip faces
	__shared__ r3d_int face_voxels[2*THREADS_PER_SM];
	__shared__ r3d_int face_tets[2*THREADS_PER_SM];

	// working vars
	cur3d_element tet;
	r3d_dvec3 vmin, vmax, vn; // voxel index range
	r3d_int voff, vflat, i, j, k, f, ii, jj, kk; // counters and such
	r3d_int tid; // local (shared memory) tet id
	r3d_int gid; // global tet id
	r3d_int vid; // local (shared memory) voxel id
	r3d_real tetvol;
	r3d_rvec3 gpt;
	unsigned char orcmp, andcmp, fflags;
	r3d_int bufind;

	r3d_plane faces[4];


	// STEP 1
	// count voxels per tet 
	// TODO: assumes that the total tet batch is <= GPU threads
	tid = threadIdx.x;
 	gid = blockIdx.x*blockDim.x + tid;
	// get voxel range and count the number of voxels to be processed
	numvox[tid] = 0;
	if(gid < nelem) {
		cur3du_get_aabb(elems[gid], n, d, vmin, vmax);
		numvox[tid] = (vmax.i - vmin.i)*(vmax.j - vmin.j)*(vmax.k - vmin.k); 
	}
	__syncthreads();

	// STEP 1.5
	// TODO: Better prefix sum implementation! 
	// this is an ultra-stupid serial version
	__shared__ r3d_int voxels_this_block;
	if(threadIdx.x == 0) {
		voxels_this_block = 0;
		for(i = 0; i < blockDim.x; ++i)
			voxels_this_block += numvox[i];
	}
	__syncthreads();

	// STEP 2
	// process all voxels in the AABBs and bin into separate buffers
	// for face and interior voxels
	// each thread gets one voxel
	__shared__ r3d_int tid_last; // last voxels and tids for the last thread in each loop
	__shared__ r3d_int voff_last; // last voxels and tids for the last thread in each loop

	// absolute indices in the voxel ring buffer 
	__shared__ r3d_int vbuf_start; 
	__shared__ r3d_int vbuf_end; 

	if(threadIdx.x == 0) {
		tid_last = 0;
		voff_last = 0;

		vbuf_start = 0;
		vbuf_end = 0;
	}

	__syncthreads();

	for(vid = threadIdx.x; vid < voxels_this_block; vid += blockDim.x) {

		// short naive prefix scan to get voxel offset
		// TODO: Faster implementation!
		// if a prefix scan over all shared-memory indices 
		// is done, can use binary search here... is it even worth it?
		tid = tid_last; // this voxel's tet, in the local array 
		voff = voff_last; // offset of this tet in voxels
		while(voff + numvox[tid] < vid)
			voff += numvox[tid++];

		// save the last voxel for faster scan in the next loop
		if(threadIdx.x == blockDim.x - 1) {
			tid_last = tid;
			voff_last = voff;	
		}
		
	 	gid = blockIdx.x*blockDim.x + tid; // global array
		tet = elems[gid];

		// recompute the AABB for this tet	
		cur3du_get_aabb(tet, n, d, vmin, vmax);
		vn.i = vmax.i - vmin.i;
		vn.j = vmax.j - vmin.j;
		vn.k = vmax.k - vmin.k;
	
		// get the grid index of this voxel
		vflat = vid - voff; 
		i = vflat/(vn.j*vn.k); 
		j = (vflat - vn.j*vn.k*i)/vn.k;
		k = vflat - vn.j*vn.k*i - vn.k*j;
		i += vmin.i; j += vmin.j; k += vmin.k;

		// TODO: Put this dedicated AABB computation into a dedicated function

		// properly orient the tet
		tetvol = cur3du_orient(tet.pos[0], tet.pos[1], tet.pos[2], tet.pos[3]);
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
			gpt.x = (ii + i)*d.x; gpt.y = (jj + j)*d.y; gpt.z = (kk + k)*d.z;
			fflags = 0x00;
			for(f = 0; f < 4; ++f) 
				if(faces[f].d + dot(gpt, faces[f].n) > 0.0) fflags |= (1 << f);
			andcmp &= fflags;
			orcmp |= fflags;
		}

		r3d_int tag_face = 0;

		// handle the appropriate voxel types
		if(andcmp == 0x0f) 
			atomicAdd(&rho[n.j*n.k*i + n.k*j + k], tet.mass/(tetvol + 1.0e-99));
		else if(orcmp == 0x0f) tag_face = 1;
		__syncthreads();

		// STEP 3
		// accumulate face voxels to a ring buffer
		
		// use prefix scan in shared memory to get the buffer indices for parallel write
		pout = 0;
		pin = 1;
		cuminds[threadIdx.x + 1] = tag_face; 
		if(threadIdx.x == 0) cuminds[0] = 0;
		for (offset = 1; offset < blockDim.x; offset *= 2) {
		    pout = 1 - pout;
		    pin  = 1 - pout;
		    __syncthreads();
		    cuminds[pout*blockDim.x + threadIdx.x] = cuminds[pin*blockDim.x + threadIdx.x];
		    if(threadIdx.x >= offset)
		        cuminds[pout*blockDim.x + threadIdx.x] += cuminds[pin*blockDim.x + threadIdx.x - offset];
		}
		__syncthreads();
		bufind = cuminds[pout*blockDim.x + threadIdx.x];

		// parallel write to the ring buffer
		if(tag_face) {
			face_voxels[(vbuf_end + bufind)%(2*THREADS_PER_SM)] = n.j*n.k*i + n.k*j + k;
			face_tets[(vbuf_end + bufind)%(2*THREADS_PER_SM)] = tid;
		}
		if(threadIdx.x == blockDim.x - 1) {
			vbuf_end += bufind + tag_face;
		}
		__syncthreads();

		// finally, parallel reduction of face voxels (1 per thread)
		if(vbuf_end - vbuf_start > THREADS_PER_SM) {

			// recompute i, j, k, faces for this voxel
			vflat = face_voxels[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]; 
			i = vflat/(n.j*n.k); 
			j = (vflat - n.j*n.k*i)/n.k;
			k = vflat - n.j*n.k*i - n.k*j;
			tet = elems[blockIdx.x*blockDim.x + face_tets[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]];

			// clip and reduce to grid
			atomicAdd(&rho[vflat], cur3d_clip_and_reduce(tet, i, j, k, d));

			// shift ring buffer head
			vbuf_start += THREADS_PER_SM;
		} 

		__syncthreads();
	}

	// clean up any face voxels remaining in the ring buffer
	if(threadIdx.x < vbuf_end - vbuf_start) {

		// recompute i, j, k, faces for this voxel
		vflat = face_voxels[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]; 
		i = vflat/(n.j*n.k); 
		j = (vflat - n.j*n.k*i)/n.k;
		k = vflat - n.j*n.k*i - n.k*j;
		tet = elems[blockIdx.x*blockDim.x + face_tets[(threadIdx.x + vbuf_start)%(2*THREADS_PER_SM)]];

		// clip and reduce to grid
		atomicAdd(&rho[vflat], cur3d_clip_and_reduce(tet, i, j, k, d));
	}
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

__device__ r3d_real cur3d_clip_and_reduce(cur3d_element tet, r3d_int i, r3d_int j, r3d_int k, r3d_rvec3 d) {

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

	tetvol = cur3du_orient(tet.pos[0], tet.pos[1], tet.pos[2], tet.pos[3]);
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
		gpt.x = (ii + i)*d.x; gpt.y = (jj + j)*d.y; gpt.z = (kk + k)*d.z;
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
	cur3d_clip_tet(&poly, andcmp);
	cur3d_reduce(&poly, 0, moments);

	return tet.mass/(tetvol + 1.0e-99)*moments[0]/(d.x*d.y*d.z);

}


__device__ void cur3d_clip_tet(r3d_poly* poly, unsigned char andcmp) {

	// variable declarations
	r3d_int nvstack;
	unsigned char vstack[4*R3D_MAX_VERTS];
	unsigned char v, f, ff, np, vcur, vprev, firstnewvert, prevnewvert;
	unsigned char fmask, ffmask;

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
			
	for(f = 0; f < 4; ++f) {

		fmask = (1 << f);
		if(andcmp & fmask) continue;

		// find the first vertex lying outside of the face
		// only need to find one (taking advantage of convexity)
		vcur = R3D_MAX_VERTS;
		for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
			if(!(vertbuffer[v].orient.fflags & (CLIP_MASK | fmask))) vcur = v;
		if(vcur == R3D_MAX_VERTS) continue;
		
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

			// pop the stack
			vcur = vstack[--nvstack];
			vprev = vstack[--nvstack];

			// if the vertex has already been clipped, ignore it
			if(vertbuffer[vcur].orient.fflags & CLIP_MASK) continue; 

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

}

__device__ void cur3d_reduce(r3d_poly* poly, r3d_int polyorder, r3d_real* moments) {

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

	// zero the moments
	for(m = 0; m < 10; ++m)
		moments[m] = 0.0;

	// find the first unclipped vertex
	vcur = R3D_MAX_VERTS;
	for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
		if(!(vertbuffer[v].orient.fflags & CLIP_MASK)) vcur = v;
	
	// return if all vertices have been clipped
	if(vcur == R3D_MAX_VERTS) return;

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

__device__ r3d_real cur3du_orient(r3d_rvec3 pa, r3d_rvec3 pb, r3d_rvec3 pc, r3d_rvec3 pd) {
	r3d_real adx, bdx, cdx;
	r3d_real ady, bdy, cdy;
	r3d_real adz, bdz, cdz;
	adx = pa.x - pd.x;
	bdx = pb.x - pd.x;
	cdx = pc.x - pd.x;
	ady = pa.y - pd.y;
	bdy = pb.y - pd.y;
	cdy = pc.y - pd.y;
	adz = pa.z - pd.z;
	bdz = pb.z - pd.z;
	cdz = pc.z - pd.z;
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

