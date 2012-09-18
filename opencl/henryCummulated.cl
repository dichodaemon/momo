#pragma OPENCL EXTENSION cl_khr_fp64: enable

#include "doubleFunctions.cl"

__kernel void cummulated1(
  uint width, uint height,
  __constant int2 * origin, uint h, 
  __constant int2 * directions, 
  __global double * forward,
  __global double * backward,
  __global double * costs,
  __global double * cummulated
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  double value = 0;

  int dx = origin->x - column;
  int dy = origin->y - row;

  if ( dx * dx + dy * dy <= h * h ) {
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
      if ( c >= 0 && c < width && r >= 0 && r < height ) {
        int priorIndex = d_forward * width * height + r * width + c;
        value += forward[priorIndex] * t_exp( -costs[stateIndex] ) * backward[stateIndex];
      }
    }
  }
  cummulated[stateIndex] = value;
}

__kernel void cummulated2(
  uint width, uint height, uint featureLength,
  __global double * cummulated,
  __global double * features
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex   = direction * width * height + row * width + column;
  unsigned int base = direction * width * height * featureLength 
                    + row * width * featureLength 
                    + column * featureLength;
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] *= cummulated[stateIndex];
  }
}
