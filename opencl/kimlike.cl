#pragma OPENCL EXTENSION cl_khr_fp64: enable

void computeFeature( 
  float2 position, float2 velocity, float2 direction, float radius,
  uint frameSize, __constant float4 * frame, 
  __constant float * densities, __constant float * speeds, __constant float * angles,
  float * feature 
) {
  int density = 0;
  float2 avgVelocity = (float2)( 0.0, 0.0 );

  for ( int i = 0; i < 9; i++ ) {
    feature[i] = 0.;
  }
  for ( int i = 0; i < frameSize; i++ ) {
    float2 other = frame[i].lo;
    float2 diff  = position - other;
    float dist   = length( diff );
    if ( dist < radius ) {
      density += 1;
      avgVelocity += frame[i].hi;
    }
  }


  if ( density > 0 ) {
    avgVelocity /= density;
    avgVelocity = velocity - avgVelocity;

    int idx = 0;
    for ( int i = 0; i < 3; ++i ) {
      if ( density >= densities[i] ) {
        idx = i;
      }
    }
    feature[idx] = 1;

    float speed = length( avgVelocity );
    idx = 3;
    for ( int i = 0; i < 3; ++i ) {
      if ( speed >= speeds[i] ) {
        idx = i + 3;
      }
    }
    feature[idx] = 1;

    float cosine = dot( normalize( avgVelocity ), direction );
    idx = 6;
    for ( int i = 0; i < 3; i++ ) {
      if ( cosine >= angles[i] ) {
        idx = i + 6;
      }
    }
    feature[idx] = 1;
  }
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
  float f[9];
  
  computeFeature( position, velocity, dir, radius, frameSize, frame, densities, speeds, angles, f );

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;
             
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] = f[i];
  }
}
