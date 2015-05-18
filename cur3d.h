
#ifndef _CUR3D_H_
#define _CUR3D_H_

extern "C" {
#include "r3d.h"
}

// Constants
#define NUM_SM 14 // num. streaming multiprocessors
#define THREADS_PER_SM 512 // num threads to launch per SM in coarse binning

//#define CUR3D_ELEM_VERTS 4
typedef struct {
	r3d_rvec3 pos[4];
	//r3d_plane faces[4];
	r3d_real mass;
	//r3d_real rho;
} cur3d_element;

void cur3d_voxelize_elements(cur3d_element* elems_h, r3d_int nelem, r3d_real* rho_h, r3d_dvec3 n, r3d_rvec3 d);

#endif // _CUR3D_H_
