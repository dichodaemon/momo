import numpy as np
import momo

def compute_expectations( 
  feature_module, convert, radius, states, frames, w, h, 
  compute_costs, planner, compute_features, accum  
):

  momo.tick( "Compute observed" )
  momo.tick( "Discretize" )
  l = len( states )
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states[:h]]
  momo.tack( "Discretize" )
  momo.tick( "Compute" )
  mu_observed = momo.irl.features.feature_sum( 
    feature_module, 
    repr_path,
    frames 
  )
  momo.tack( "Compute" )
  momo.tack( "Compute observed" )

  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  momo.tick( "Forward-backward" )
  forward, backward, costs = planner( states[0], states[-1], avg_velocity, frames[0], w )
  momo.tack( "Forward-backward" )
  momo.tick( "Compute features" )
  features = compute_features( avg_velocity, frames[0] )
  momo.tack( "Compute features" )


  momo.tick( "Accum" )
  cummulated, w_features  = accum( forward, backward, costs, features, grid_path[0], h )
  momo.tack( "Accum" )

  mu_expected = np.sum( w_features, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )


  return mu_observed, mu_expected, cummulated, costs, grid_path


