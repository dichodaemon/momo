import numpy as np
import pylab as pl
import momo
import cvxopt
import cvxopt.solvers
import cvxopt
from cvxopt import solvers
from math import *

def learn( feature_module, convert, frame_data, ids ):
  feature_length = feature_module.FEATURE_LENGTH

  # Initialize weight vector
  w  = np.random.rand( feature_length )
  w /= np.linalg.norm( w )

  # Compute observed feature sum for selected samples
  mu_observed = w * 0.
  for o_id in ids:
    val = feature_sum( 
      feature_module, 
      [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in frame_data[o_id]["states"]], 
      frame_data[o_id]["frames"] 
    )
    mu_observed += val
  #mu_observed /= np.sum( mu_observed[:4] )

  # Main optimization loop
  mu_planned = []
  weights = []
  counts = []
  j = 0

  for reps in xrange( 18 ):
    temp_sum = w * 0.
    for o_id in ids:
      val = compute_plan_features( feature_module, convert, w, frame_data[o_id] )
      temp_sum += val
    #temp_sum /= np.sum( temp_sum[:4] )

    mu_planned.append( temp_sum )
    weights.append( w )
    counts.append( np.linalg.norm( temp_sum - mu_observed ) )
    w, x = optimize1( j, w, mu_planned, mu_observed )
    norm = np.linalg.norm( w )
    diff = w - w / norm

    r = w * 0.
    for i in xrange( len( mu_planned ) ):
      r += mu_planned[i] * x[i]

    print "w = mu - mu_e", w - ( -r - mu_observed )

    w = w / norm
    j += 1
    print "norm", norm
    print "x", x
    print "costs", counts
    if norm < 1E-4:
      break

  w = weights[np.argmin( counts )]
  return w


def compute_plan_features( feature_module, convert, w, data ):
  compute_costs = momo.irl.features.flow.compute_costs( convert, radius = 3 )
  plan = momo.planning.dijkstra()

  states = data["states"]
  frames = data["frames"]
  start = states[0]
  goal = states[-1]
  goal = convert.from_world2( goal )
  current = convert.from_world2( start )
  traversed = []
  count = 0

  result = None

  if True:
    pl.figure( 1, figsize = ( 30, 10 ), dpi = 75 )
    pl.ion()

  while True:
    if count >= len( states ):
      count = len( states ) - 1
    velocity = np.linalg.norm( states[count][2:] )
    costs = compute_costs( velocity, w, frames[count] )

    cummulated, parents = plan( costs, goal )
    path = plan.get_path( parents, current )


    traversed.append( path[0] )
    current = path[1]
    converted = convert.to_world2( current, velocity )
    
    if result == None:
      result  = feature_module.compute_feature( converted, frames[count] )
    else:
      result += feature_module.compute_feature( converted, frames[count] )

    if path.shape[0] == 2:
      break

    count += 1

  if True:
    pl.clf()
    pl.axis( "scaled" )
    pl.xlim( convert.x, convert.x2 )
    pl.ylim( convert.y, convert.y2 )
    pl.imshow( cummulated[0], pl.cm.jet, None, None, "none", 
      origin = "upper", extent = (convert.x, convert.x2, convert.y, convert.y2), 
      vmin = 0, vmax = 100
    )

    pl.plot( 
      [ v[0] for v in states[:]],
      [ v[1] for v in states[:]],
      "w."
    )

    pl.plot( 
      [ convert.to_world( v )[0] for v in traversed[:]],
      [ convert.to_world( v )[1] for v in traversed[:]],
      "c."
    )

    pl.plot( 
      [ convert.to_world( v )[0] for v in path[:]],
      [ convert.to_world( v )[1] for v in path[:]],
      "m."
    )

    pl.draw()

  return result

def feature_sum( feature_module, states, frames ):
  result = np.array( [0.] * feature_module.FEATURE_LENGTH )
  for i in xrange( len( states ) ):
    result += feature_module.compute_feature( states[i], frames[i] )
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

def optimize1(  j, w, mu_planned, mu_observed ):
  n = len( w ) + j + 1
  p = cvxopt.matrix( np.zeros( ( n, n ) ), tc = "d" )
  q = cvxopt.matrix( np.zeros( n ), tc = "d" )
  for i in xrange( len( w ) ):
    p[i, i] = 1.0
  a = cvxopt.matrix( np.zeros( ( len( w ) + 1, n ) ), tc = "d" )
  for i in xrange( len( w ), n ):
    a[0, i] = 1
  for i in xrange( len( w ) ):
    a[1 + i, i] = 1
    for tj in xrange( j + 1 ):
      a[1 + i, len( w ) + tj] = mu_planned[tj][i] 
  b = cvxopt.matrix( np.ones( 1 + len( w ) ), tc = "d" )
  for i in xrange( len( w ) ):
    b[1 + i] = -mu_observed[i]
  g = cvxopt.matrix( np.zeros( ( n , n ) ), tc = "d" )
  for i in xrange( n ):
    g[i, i] = 1
  h = cvxopt.matrix( np.zeros( n ), tc = "d" )
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
