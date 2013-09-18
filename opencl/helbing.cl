#pragma OPENCL EXTENSION cl_khr_fp64: enable

uint maxIdx( float value, __constant float * reference, uint length )
{
  uint result = 0;
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
  float lambda, __constant float * angles,
  float * feature 
) {
  for ( int i = 0; i < 3; i++ ) {
    feature[i] = 0.;
  }
  for ( int i = 0; i < frameSize; i++ ) {
    float2 otherX = frame[i].lo;
    float2 otherV = frame[i].hi;

    float2 xRel  = otherX - position;
    float xLen   = length( xRel );

    float2 n = normalize( xRel );
    float2 e = normalize( otherV );
    float cosPhi = dot( -n, e );  
    float force  = ( lambda + 0.5 * ( 1 - lambda ) * ( 1 + cosPhi ) ) * exp( ( 2 * radius - xLen ) );
    if ( force > 0.5 ) {
      feature[maxIdx( cosPhi, angles, 3 )] += 1;
    }
  }
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  __constant float2 * directions,
  uint width, uint height, uint featureLength,
  uint frameSize, __constant float4 * frame, 
  float lambda, __constant float * angles,
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir      = normalize( directions[direction] );
  float2 velocity = dir * speed; 
  float2 position = (float2)( column * delta, row * delta );
  float f[3];
  
  computeFeature( position, velocity, radius, frameSize, frame, lambda, angles, f );

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;
             
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] = f[i];
  }
}
