#pragma OPENCL EXTENSION cl_khr_fp64: enable

#include "doubleFunctions.cl"


__kernel void forwardPass(
  uint width, uint height,
  __constant int2 * directions, 
  __global double * costs,
  __global int * fMasks,
  __global int * bMasks,
  __global double * f1, __global double * f2,
  __global double * b1, __global double * b2
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  int fMask = fMasks[stateIndex];
  int bMask = bMasks[stateIndex];

  double fValue = 0;
  double bValue = 0;

  for ( int d = direction - 1; d < direction + 2; d++ ) {
    int dForward = d;
    if ( dForward < 0 ) {
      dForward += 8;
    } else if ( dForward > 7 ) {
      dForward -= 8;
    }
    // Forward
    int2 delta = directions[direction];

    int c = column - delta.x;
    int r = row - delta.y;
    if ( c >= 0 && c < width && r >= 0 && r < height ) {
      int priorIndex = dForward * width * height + r * width + c;
      fValue += fMask;
      // TODO: Check t_exp is necessary
      fValue += f1[priorIndex] * exp( -costs[stateIndex] );
    }

    // Backward
    delta = directions[dForward];

    c = column + delta.x;
    r = row + delta.y;
    if ( c >= 0 && c < width && r >= 0 && r < height ) {
      int nextIndex = dForward * width * height + r * width + c;
      bValue += bMask;
      // TODO: Check t_exp is necessary
      bValue += b1[nextIndex] * exp( -costs[nextIndex] );
    }
  }
  f2[stateIndex] = fValue;
  b2[stateIndex] = bValue;
}

__kernel void updatePass(
  uint width, uint height,
  __global double * f1, __global double * f2,
  __global double * b1, __global double * b2
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  f1[stateIndex] = f2[stateIndex];
  b1[stateIndex] = b2[stateIndex];
}

