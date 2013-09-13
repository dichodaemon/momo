import numpy as np
import momo

def compute_expectations( 
  states, frames, w, h, 
  convert, compute_costs, planner, compute_features, accum  
):
  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )
  features = compute_features( avg_velocity, frames[0] )

  momo.tick( "Forward-backward" )
  forward, backward, costs = planner( states[0], states[-1], features, w, avg_velocity )
  momo.tack( "Forward-backward" )

  momo.tick( "Accum" )
  cummulated, w_features  = accum( 
    forward, backward, costs, features, 
    convert.from_world2( states[0] ), h 
  )
  momo.tack( "Accum" )

  mu_expected = np.sum( w_features, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )

  return mu_expected, cummulated, costs


