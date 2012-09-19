// Based on the implementation of Dan Ginsburg
// http://code.google.com/p/opencl-book-samples/


__kernel  void dijkstraPass1(
  uint width, uint height, 
  __constant int2 * directions, 
  __global float * costArray, 
  __global int * maskArray, 
  __global float * cummulated, 
  __global float * tmpCummulated,
  __global int * tmpParents
) {
  int direction = get_global_id( 0 );
  int row       = get_global_id( 1 );
  int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  if ( maskArray[stateIndex] != 0 ) {
    maskArray[stateIndex] = 0;
    for ( int d = direction - 1; d < direction + 2; d++ ) {
      int dd = d;
      if ( dd < 0 ) {
        dd += 8;
      } else if ( dd > 7 ) {
        dd -= 8;
      }
      int2 delta = directions[direction];
      int c = column - delta.x;
      int r = row - delta.y;
      if ( c >= 0 and c < width && r >= 0 && r < height ) {
        int newIndex = dd * width * height + r * width + c;
        float g = cummulated[stateIndex] + costArray[stateIndex];
        if ( tmpCummulated[newIndex] > g ) {
          tmpCummulated[newIndex] = g;
          tmpParents[newIndex] = stateIndex;
        }
      }
    }
  }
}

__kernel  void dijkstraPass2(
  uint width, uint height, 
  __global int *maskArray, 
  __global float *cummulated, 
  __global float *tmpCummulated,
  __global int * parents,
  __global int *tmpParents
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;

  if ( cummulated[stateIndex] > tmpCummulated[stateIndex] ) {
    cummulated[stateIndex] = tmpCummulated[stateIndex];
    parents[stateIndex] = tmpParents[stateIndex];
    maskArray[stateIndex] = 1;
  }
  tmpCummulated[stateIndex] = cummulated[stateIndex];
}

__kernel void initializeBuffers( 
  uint width, uint height, 
  __global int * maskArray, 
  __global float * cummulated, 
  __global float * tmpCummulated,
  __global int * parents, 
  __global int * tmpParents,
  __constant int3 * sourceVertex 
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  unsigned int sourceIndex = sourceVertex->z * width * height + sourceVertex->y * width + sourceVertex->x;
  unsigned int stateIndex  = direction * width * height + row * width + column;


  if ( sourceIndex == stateIndex ) {
    maskArray[stateIndex] = 1;
    cummulated[stateIndex] = 0.0;
    tmpCummulated[stateIndex] = 0.0;
  } else {
    maskArray[stateIndex] = 0;
    cummulated[stateIndex] = FLT_MAX;
    tmpCummulated[stateIndex] = FLT_MAX;
  }
  parents[stateIndex] = -1;
  tmpParents[stateIndex] = -1;
}

