import numpy as np
import momo
from compute_cummulated import *

def compute_expectations( feature_module, convert, radius, states, frames, w, h ):
  compute_costs = feature_module.compute_costs( convert, radius )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )

  l = len( states )
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states[:h]]
  mu_observed = momo.irl.features.feature_sum( 
    feature_module, 
    repr_path[:h],
    frames[:h] 
  )

  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  forward, backward, costs = planner( states[0], states[-1], avg_velocity, frames[0], w )
  features = compute_features( avg_velocity, frames[0] )


  accum = compute_cummulated()
  cummulated, w_features  = accum( forward, backward, costs, features, grid_path[0], h )

  mu_expected = np.sum( w_features, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )


  return mu_observed, mu_expected, cummulated, costs, grid_path


