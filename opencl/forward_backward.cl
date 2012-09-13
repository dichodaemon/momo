__kernel void computeForward1(
  uint width, uint height,
  __constant int2 * directions, 
  __global float * costs,
  __global int * m1, global int * m2, 
  __global float f1, __global float * f2
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  int mask = m1[stateIndex];

  float origin = 0;
  if ( mask == 1 ) {
    origin = 1;
  }

  if ( mask > 0 ) {
    origin += f1[stateIndex];
    for ( int d = direction - 1; d < direction + 2; d++ ) {
      int dd = d;
      if ( dd < 0 ) {
        dd += 8;
      } else if ( dd > 7 ) {
        dd -= 8;
      }
      int2 delta = directions[dd];

      // Update this state
      int c = column - delta.x;
      int r = row - delta.y;
      if ( c >= 0 and c < width && r >= 0 && r < height ) {
        int priorIndex = dd * width * height + r * width + c;
        int priorMask = m1[priorIndex];
        if ( priorMask > 0 ) {
          origin += f1[priorIndex] * exp( -costs[priorIndex] );
        }
      }

      // Enable next states
      c = column + delta.x;
      r = row + delta.y;
      if ( c >= 0 and c < width && r >= 0 && r < height ) {
        int nextIndex = dd * width * height + r * width + c;
        int nextMask = m1[priorIndex];
        if ( nextMask == 0 ) {
          m2[priorIndex] = 3;
        }
      }
    }
    f2[stateIndex] = origin;
    m2[stateIndex] = mask;
  }
}

__kernel void computeForward2(
  uint width, uint height,
  __global int * m1, global int * m2, 
  __global float f1, __global float * f2
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  f1[stateIndex] = f2[stateIndex];
  m1[stateIndex] = m2[stateIndex];
}

