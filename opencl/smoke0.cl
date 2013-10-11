#pragma OPENCL EXTENSION cl_khr_fp64: enable

void computeFeature( 
  float2 position, float2 velocity, float radius,
  uint frameSize, __constant float4 * frame, 
  float * feature 
) {
  for ( int i = 0; i < 5; i++ ) {
    feature[i] = 0.;
  }
  for ( int i = 0; i < frameSize; i++ ) {
    float2 otherX = frame[i].lo;
    float2 otherV = frame[i].hi;

    float2 xRel  = otherX - position;
    float xLen   = length( xRel );

    feature[0] = xLen;
    feature[1] = velocity.x;
    feature[2] = velocity.y;
    feature[3] = position.x;
    feature[4] = position.y;
  }
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  __constant float2 * directions,
  uint width, uint height, uint featureLength,
  uint frameSize, __constant float4 * frame, 
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir      = normalize( directions[direction] );
  float2 velocity = dir * speed; 
  float2 position = (float2)( column * delta, row * delta );
  float f[5];
  
  computeFeature( position, velocity, radius, frameSize, frame, f );

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;
             
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] = f[i];
  }
}
