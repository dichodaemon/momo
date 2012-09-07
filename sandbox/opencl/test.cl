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
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  unsigned int stateIndex = direction * width * height + row * width + column;
  /*if ( maskArray[stateIndex] == 1 ) {*/
    /*maskArray[stateIndex] = 0;*/
    /*if ( width - column > 1 && cummulated[stateIndex + 1] == 0 ) {*/
      /*maskArray[stateIndex + 1] = 1;*/
      /*tmpCummulated[stateIndex + 1] = row;*/
    /*} */
    /*if ( height - row > 1 && cummulated[stateIndex + width] == 0 ) {*/
      /*maskArray[stateIndex + width] = 1;*/
      /*tmpCummulated[stateIndex + width] = row + 1;*/
    /*}*/
  /*}*/

}

__kernel  void dijkstraPass2(
  uint width, uint height, 
  __global int *maskArray, 
  __global float *cummulated, 
  __global float *tmpCummulated,
  __global int * parents,
  __global int *tmpParents
) {
  /*unsigned int direction = get_global_id( 0 );*/
  /*unsigned int row       = get_global_id( 1 );*/
  /*unsigned int column    = get_global_id( 2 );*/

  /*unsigned int stateIndex = direction * width * height + row * width + column;*/

  /*cummulated[stateIndex] = tmpCummulated[stateIndex];*/
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

  unsigned int sourceIndex = sourceVertex->z * width * height + sourceVertex->x * width + sourceVertex->y;
  unsigned int stateIndex  = direction * width * height + row * width + column;

  maskArray[stateIndex] = 0;
  if ( sourceIndex == stateIndex ) {
    cummulated[stateIndex] = 10;
  } else {
    cummulated[stateIndex] = 0;
  }
}

