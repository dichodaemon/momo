import numpy as np
import momo

def learn( feature_module, convert, frame_data, ids, radius, replan ):
  feature_length = feature_module.FEATURE_LENGTH
  
  compute_costs = feature_module.compute_costs( convert, radius )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )

  # Initialize weight vector
  w  = np.random.rand( feature_length )
  w /= np.linalg.norm( w )

  for o_id in ids:
    gradient = compute_gradient( 
      feature_module, planner, convert, compute_features,
      frame_data[o_id]["states"], frame_data[o_id]["frames"],
      w, replan
    )
    for i in xrange( feature_length ):
      w[i] *= exp( -gamma * gradient[i] )

  return w

def compute_gradient( feature_module, planner, convert, compute_features, states, frames, w, replan ):
  mu_observed = momo.irl.features.feature_sum( 
    feature_module, 
    [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states], 
    frames 
  )

  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  forward, backward, costs = planner( states[0], states[-1], avg_velocity, frames[0], w )
  features = compute_features( avg_velocity, frames[0] )

  mu_expected = np.zeros( feature_module.FEATURE_LENGTH )
  
  for d in xrange( forward.shape[0] ):
    for y in xrange( forward.shape[1] ):
      for x in xrange( forward.shape[2] ):
        for d1 in xrange( d - 1, d + 2 ):
          if d1 > 7:
            d1 -= 4
          elif d1 < 0:
            d1 += 4

          x1 = x + feature_module.DIRECTIONS[d1][0]
          y1 = y + feature_module.DIRECTIONS[d1][1]
          
          #if x1 >= 0 and x1 < forward.shape[2] and y1 >= 0 and y1 < forward.shape[1]:
            #mu_expected += features[d1, y1, x1] * forward[d, y, x] * costs[d1, y1, x1] * backward[d1, y1, x1]  

  start = convert.from_world2( states[0] )
  goal  = convert.from_world2( states[-1] )

  print "forward", forward[tuple( reversed( goal.tolist() ) )]
  print "backward", backward[tuple( reversed( start.tolist() ) )]

  return mu_observed - mu_expected

