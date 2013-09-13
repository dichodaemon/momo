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
  __constant float * densities, __constant float * speeds, __constant float * angles,
  float * feature 
) {
  int density = 0;
  float angSum = 0.0;
  float magSum = 0.0;

  for ( int i = 0; i < 10; i++ ) {
    feature[i] = 0.;
  }
  for ( int i = 0; i < frameSize; i++ ) {
    float2 other = frame[i].lo;
    float2 xRel  = other - position;
    float xLen   = length( xRel );
    if ( xLen < radius ) {
      density += 1;
      float2 vRel = frame[i].hi - velocity;
      float vLen = length( vRel );
      float a = dot( vRel / vLen, xLen / xLen );
      angSum += a;
      magSum += vLen;
    }
  }



  if ( density > 0 ) {
    feature[maxIdx( density, densities, 3)] = 1;

    float speed = magSum / density;
    feature[3 + maxIdx( speed, speeds, 3 )] = 1;

    float cosine = angSum / density;
    feature[6 + maxIdx( cosine, angles, 3 )] = 1;
  }
  feature[9] = 1;
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  __constant float2 * directions,
  uint width, uint height, uint featureLength,
  uint frameSize, __constant float4 * frame, 
  __constant float * densities, __constant float * speeds, __constant float * angles,
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 dir      = normalize( directions[direction] );
  float2 velocity = dir * speed; 
  float2 position = (float2)( column * delta, row * delta );
  float f[10];
  
  computeFeature( position, velocity, radius, frameSize, frame, densities, speeds, angles, f );

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;
             
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] = f[i];
  }
}
