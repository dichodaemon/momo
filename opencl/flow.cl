//#pragma OPENCL EXTENSION cl_khr_fp64: enable

void computeFeature( 
  float2 position, float2 velocity, float radius,
  uint frameSize, __constant float4 * frame, 
  __constant float2 * angles, __constant float * speeds, 
  float * feature 
) {
  int density = 0;
  float2 avgVelocity = (float2)( 0.0, 0.0 );
  for ( int i = 0; i < 18; i++ ) {
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
  if ( density >= 3 ) {
    feature[3] = 1;
  } else {
    feature[density] = 1;
  }
  if ( density == 0 ) {
    feature[16] = 1;
  } else {
    avgVelocity /= density;
    avgVelocity  = velocity - avgVelocity;
    avgVelocity.y = fabs( avgVelocity.y );
    float2 angle    = normalize( avgVelocity );
    uint angleIndex = 0;
    float maxDot  = -1;
    for ( int i = 0; i < 4; i++ ) {
      float a = dot( angles[i], angle );
      if ( a > maxDot ) {
        maxDot     = a;
        angleIndex = i;
      }
    }
    float speed = length( avgVelocity );
    uint speedIndex = 0;
    for ( int i = 0; i < 3; i++ ) {
      if ( speed >= speeds[i] ) {
        speedIndex = i;
      }
    }
    feature[4 + angleIndex * 3 + speedIndex] = 1;
  }
  feature[17] = 1;
}

__kernel void computeCosts( 
  float speed, float delta, float radius, 
  uint width, uint height,
  uint frameSize, __constant float4 * frame, 
  __constant float2 * directions, __constant float2 * angles, __constant float * speeds, 
  __constant float * theta, __global float * costs
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 position = (float2)( column * delta, row * delta );
  float2 velocity = directions[direction] * speed;
  float f[18];
  
  computeFeature( position, velocity, radius, frameSize, frame, angles, speeds, f );

  float cost = 0;
  for ( int i = 0; i < 18; i++ ) {
    cost += f[i] * theta[i];// * length( directions[direction] );
  }
  costs[direction * width * height + row * width + column] = cost;
}

__kernel void computeFeatures( 
  float speed, float delta, float radius, 
  uint width, uint height, uint featureLength,
  uint frameSize, __constant float4 * frame, 
  __constant float2 * directions, __constant float2 * angles, __constant float * speeds, 
  __global float * features
) {
  unsigned int direction = get_global_id( 0 );
  unsigned int row       = get_global_id( 1 );
  unsigned int column    = get_global_id( 2 );

  float2 position = (float2)( column * delta, row * delta );
  float2 velocity = directions[direction] * speed;
  float f[18];
  
  computeFeature( position, velocity, radius, frameSize, frame, angles, speeds, f );

  int base =  direction * width * height * featureLength 
            + row * width * featureLength
            + column * featureLength;
             
  for ( int i = 0; i < featureLength; i++ ) {
    features[base + i] = f[i];
  }
}
