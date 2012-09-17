import numpy as np
import pylab as pl
import momo
import cvxopt
import cvxopt.solvers
import cvxopt
from cvxopt import solvers
from math import *

def learn( feature_module, convert, frame_data, ids, radius, replan ):
  feature_length = feature_module.FEATURE_LENGTH

  compute_costs = feature_module.compute_costs( convert, radius )
  planner = momo.irl.planning.dijkstra( convert, compute_costs )

  # Initialize weight vector
  w  = np.random.rand( feature_length )
  w /= np.linalg.norm( w )

  # Compute observed feature sum for selected samples
  mu_observed = w * 0.
  for o_id in ids:
    val = momo.irl.features.feature_sum( 
      feature_module, 
      [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in frame_data[o_id]["states"]], 
      frame_data[o_id]["frames"] 
    )
    mu_observed += val

  # Main optimization loop
  mu_planned = []
  weights = []
  counts = []
  j = 0

  for reps in xrange( 18 ):
    temp_sum = w * 0.
    for o_id in ids:
      val = compute_plan_features( feature_module, convert, planner, w, frame_data[o_id], radius, replan )
      temp_sum += val

    mu_planned.append( temp_sum )
    weights.append( w )
    counts.append( np.linalg.norm( temp_sum - mu_observed ) )
    w, x = optimize( j, w, mu_planned, mu_observed )
    norm = np.linalg.norm( w )
    diff = w - w / norm
    w = w / norm
    j += 1
    if norm < 1E-4:
      break

  w = weights[np.argmin( counts )]
  return w


def compute_plan_features( feature_module, convert, planner, w, data, radius, replan ):
  states = data["states"]
  frames = data["frames"]

  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  path   = planner( states[0], states[-1], avg_velocity, frames, replan, w )
  result = np.sum( 
    [
      feature_module.compute_feature( 
        path[i], frames[min( i, len( frames ) )], radius 
      ) 
      for i in xrange( len( path ) )
    ], 
    0 
  )

  if True:
    pl.figure( 1, figsize = ( 30, 10 ), dpi = 75 )
    pl.ion()
    pl.clf()
    pl.axis( "scaled" )
    pl.xlim( convert.x, convert.x2 )
    pl.ylim( convert.y, convert.y2 )
    pl.plot( 
      [ v[0] for v in states[:]],
      [ v[1] for v in states[:]],
      "r."
    )

    pl.plot( 
      [ v[0] for v in path[:]],
      [ v[1] for v in path[:]],
      "b."
    )

    pl.draw()

  return result


def optimize(  j, w, mu_planned, mu_observed ):
  n = len( w ) + j + 1
  p = cvxopt.matrix( np.zeros( ( n, n ) ), tc = "d" )
  q = cvxopt.matrix( np.zeros( n ), tc = "d" )
  for i in xrange( len( w ) ):
    p[i, i] = 1.0
  a = cvxopt.matrix( np.zeros( ( 1, n ) ), tc = "d" )
  for i in xrange( len( w ), n ):
    a[0, i] = 1
  b = cvxopt.matrix( np.ones( 1 ), tc = "d" )
  g = cvxopt.matrix( np.zeros( ( n + len( w ), n ) ), tc = "d" )
  for i in xrange( n ):
    g[i, i] = 1
  for i in xrange( len( w ) ):
    g[n + i, i] = 1
    for tj in xrange( j + 1 ):
      g[n + i, len( w ) + tj] = mu_planned[tj][i] #Why does this not work with negative?
  h = cvxopt.matrix( np.zeros( n + len( w ) ), tc = "d" )
  for i in xrange( len( w ) ):
    h[n + i] = -mu_observed[i]
  solvers.options["maxiters"] = 20
  solvers.options["show_progress"] = False
  result = solvers.qp( p, q, - g, -h, a, b, "glpk" )
  r_w = w * 0.
  for i in xrange( len( w ) ):
    r_w[i] = result["x"][i]
  r_x = np.zeros( j + 1 )
  for i in xrange( j + 1 ):
    r_x[i] = result["x"][len( w ) + i]
  return r_w, r_x

