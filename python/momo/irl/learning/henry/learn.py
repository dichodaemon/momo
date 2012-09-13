import numpy as np

def learn( feature_module, convert, planner, frame_data, ids, radius, replan ):
  feature_length = feature_module.FEATURE_LENGTH
  
  # Initialize weight vector
  w  = np.random.rand( feature_length )
  w /= np.linalg.norm( w )

  for o_id in ids:
    gradient = compute_gradient( 
      feature_module, convert, radius, replan,
      frame_data[o_id]["states"], frame_data[o_id]["frames"]
    )
    for i in xrange( feature_length ):
      w[i] *= exp( -gamma * gradient[i] )

  return w

def compute_gradient( feature_module, convert, radius, replan, states, frames ):
  mu_observed = momo.irl.features.feature_sum( 
    feature_module, 
    [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states], 
    frames 
  )
  
  forward  = compute_forward( convert, radius, frames )
  backward = compute_backward( convert, radius, frames )
