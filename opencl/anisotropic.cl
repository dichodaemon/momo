#pragma OPENCL EXTENSION cl_khr_fp64: enable

uint maxIdx( float value, __constant float * reference, uint length )
{
  uint result = -1;
  for ( uint i = 0; i < length; ++i ) {
    if ( value >= reference[i] ) {
      result = i;
    }
  }
  return result;
}

void computeFeature( 
  float2 position, float2 velocity, float radius,
  uint frameSize, __constant float4 * frame, 
  float lambda,
  float * feature 
) {
  for ( int i = 0; i < 2; i++ ) {
    feature[i] = 0.;
  }
  for ( int i = 0; i < frameSize; i++ ) {
    float2 otherX = frame[i].lo;
    float2 otherV = frame[i].hi;

    float2 xRel  = otherX - position;
    float xLen   = length( xRel );
    float2 vRel = otherV - velocity;
    float vLen = length( vRel );

    float2 n = normalize( xRel );
    float2 e = normalize( vRel );
    float cosPhi1 = dot( -n, e );
    float cosPhi2 = dot( -n, normalize( otherV ) );  
    float force1 = ( 1 + cosPhi1 ) * exp( 2 * radius - xLen ) * vLen;
    float force2 = ( lambda + 0.5 * ( 1 - lambda ) * ( 1 + cosPhi2 ) ) * exp( 2 * radius - xLen );
    feature[0] += force1;
    feature[1] += force2;
  }
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  __constant float2 * directions,
  uint width, uint height, uint featureLength,
  uint frameSize, __constant float4 * frame, 
  float lambda,
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir      = normalize( directions[direction] );
  float2 velocity = dir * speed; 
  float2 position = (float2)( column * delta, row * delta );
  float f[3];
  
  computeFeature( position, velocity, radius, frameSize, frame, lambda, f );

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;
             
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] = f[i];
  }
}
