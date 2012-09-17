__kernel void computeForward1(
  uint width, uint height, int sense,
  __constant int2 * directions, 
  __global float * costs,
  __global int * m1, global int * m2, 
  __global float * f1, __global float * f2
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
      int d_forward = d;
      if ( d_forward < 0 ) {
        d_forward += 8;
      } else if ( d_forward > 7 ) {
        d_forward -= 8;
      }
      int d_backward = d_forward + 4;
      if ( d_backward > 7 ) {
        d_backward -= 8;
      } 
      if ( sense == -1 ) {
        int tmp = d_forward;
        d_forward = d_backward;
        d_backward = tmp;
      }
      int2 delta = directions[d_backward];

      // Update this state
      int c = column + delta.x;
      int r = row + delta.y;
      if ( c >= 0 and c < width && r >= 0 && r < height ) {
        int priorIndex = d_backward * width * height + r * width + c;
        int priorMask = m1[priorIndex];
        if ( priorMask > 0 ) {
          origin += f1[priorIndex] * exp( -costs[priorIndex] * 2 );
        }
      }

      // Enable next states
      delta = directions[d_forward];
      c = column + delta.x;
      r = row + delta.y;
      if ( c >= 0 and c < width && r >= 0 && r < height ) {
        int nextIndex = d_forward * width * height + r * width + c;
        int nextMask = m1[nextIndex];
        if ( nextMask == 0 ) {
          m2[nextIndex] = 3;
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
  __global float * f1, __global float * f2
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  f1[stateIndex] = f2[stateIndex];
  m1[stateIndex] = m2[stateIndex];
}

