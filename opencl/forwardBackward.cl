#pragma OPENCL EXTENSION cl_khr_fp64: enable

#include "doubleFunctions.cl"


__kernel void forwardPass(
  uint width, uint height,
  __constant int2 * directions, 
  __global double * costs,
  __global int * masks, 
  __global double * f1, __global double * f2
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  int mask = masks[stateIndex];

  double origin = 0;

  for ( int d = direction - 1; d < direction + 2; d++ ) {
    int d_forward = d;
    if ( d_forward < 0 ) {
      d_forward += 8;
    } else if ( d_forward > 7 ) {
      d_forward -= 8;
    }
    int2 delta = directions[direction];

    // Update this state
    int c = column - delta.x;
    int r = row - delta.y;
    origin += mask;
    if ( c >= 0 && c < width && r >= 0 && r < height ) {
      int priorIndex = d_forward * width * height + r * width + c;
      origin += f1[priorIndex] * t_exp( -costs[stateIndex] );
    }
  }
  f2[stateIndex] = origin;
}

__kernel void backwardPass(
  uint width, uint height, 
  __constant int2 * directions, 
  __global double * costs,
  __global int * masks, 
  __global double * f1, __global double * f2
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  int mask = masks[stateIndex];

  double origin = 0;

  for ( int d = direction - 1; d < direction + 2; d++ ) {
    int d_forward = d;
    if ( d_forward < 0 ) {
      d_forward += 8;
    } else if ( d_forward > 7 ) {
      d_forward -= 8;
    }
    int2 delta = directions[d_forward];

    // Update this state
    int c = column + delta.x;
    int r = row + delta.y;
    origin += mask;
    if ( c >= 0 && c < width && r >= 0 && r < height ) {
      int nextIndex = d_forward * width * height + r * width + c;
      origin += f1[nextIndex] * t_exp( -costs[nextIndex] );
    }
  }
  f2[stateIndex] = origin;
}

__kernel void updatePass(
  uint width, uint height,
  __global double * f1, __global double * f2
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  f1[stateIndex] = f2[stateIndex];
}

