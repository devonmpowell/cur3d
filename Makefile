#############################################################################
#
#	Makefile
#
#	libcur3d.a
#
#############################################################################

######## User options #########

# Use single-precision computations (not recommended) 
OPT += -DSINGLE_PRECISION

# Use an octree to efficiently search enclosed regions
# For most use cases, this gives negligible improvement 
#OPT += -DUSE_TREE

# Turn off clipping in voxelization routines
#OPT += -DNO_CLIPPING

# Turn off reduction in voxelization routines
#OPT += -DNO_REDUCTION

###############################

CC = /opt/cuda-6.0/bin/nvcc
CFLAGS = -I. -O3 -arch sm_20 --ptxas-options=-v --compiler-options -Wall  
SRC = cur3d.cu
DEPS = cur3d.h Makefile
OBJ = $(SRC:.cu=.o)

all: libcur3d.a

libcur3d.a: $(OBJ)
	ar -rs $@ $^

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(OPT)

clean:
	rm -rf libcur3d.a $(OBJ) 
