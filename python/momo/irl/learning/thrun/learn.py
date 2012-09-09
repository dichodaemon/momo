def learn( feature_module, frame_datai, width, height, delta, min_x, min_y ):
  feature_length = feature_module.FEATURE_LENGTH

  # Initialize weight vector
  w  = np.random.rand( feature_length )
  w /= np.linalg.norm( w )

  # Compute observed feature sum for selected samples
  mu_observed = w * 0.
  for data in frame_data.values():
    mu_observed += feature_sum( feature_module, data["states"], data["frames"] )

  # Main optimization loop
  mu_planned = []
  weights = []
  j = 0

  compute_costs = momo.irl.features.flow.compute_costs( width, height,delta )
  plan = momo.planning.dijkstra()

  while True:
    temp_sum = w * 0.
    for data in frame_data.values():
      temp_sum += compute_plan_features( feature_module, w, data, plan, compute_costs, min_x )

    mu_planned.append( temp_sum )
    w, x = optimize( j, w, mu_planned, mu_observed )
    norm = np.linalg.norm( w )
    diff = w - w / norm
    w = w / norm
    weights.append( w )

    if np.linalg.norm( diff ) < 1E-3:
      w = weights[np.argmax( x )]
      j += 1
  return w


def compute_plan_features( w, data, width, plan, compute_costs ):
  states = data["states"]
  frames = data["frames"]
  start = states[0]
  goal = states[-1]
  current = momo.grid_math.from_world( start )
  count = 0

  result = None

  while True:
    velocity = np.linalg.norm( states[count] )
    costs = compute_costs( velocity, w, frames[count] )

    path = plan( costs, current )
    current = path.pop( 0 )
    converted = momo.grid_math.to_world( current )
    
    if result == None:
      result  = feature_module.compute_feature( converted, frames[count] )
    else:
      result += feature_module.compute_feature( converted, frames[count] )

    if len( path ) == 0:
      break

    count += 1
  return result

def feature_sum( feature_module, states, frames ):
  result = np.array( [0.] * feature_module.FEATURE_LENGTH )
  for i in xrange( states ):
    result += feature_module.compute_feature( states[i], frames[i] )
  return result


def optimize(  j, w, mu_planned, mu_observed ):
  n = len( w ) + j + 1
  p = cvxopt.matrix( np.zeros( ( n, n ) ) )
  q = cvxopt.matrix( np.zeros( n ) )
  for i in xrange( len( w ) ):
    p[i, i] = 1.0
  a = cvxopt.matrix( np.zeros( ( 1, n ) ) )
  for i in xrange( len( w ), n ):
    a[0, i] = 1
  b = cvxopt.matrix( np.ones( 1 ) )
  g = cvxopt.matrix( np.zeros( ( n + len( w ), n ) ) )
  for i in xrange( n ):
    g[i, i] = 1
  for i in xrange( len( w ) ):
    g[n + i, i] = 1
    for j in xrange( j + 1 ):
      g[n + i, len( w ) + j] = -mu_planned[j][i]
  h = cvxopt.matrix( np.zeros( n + len( w ) ) )
  for i in xrange( len( w ) ):
    h[n + i] = mu_observed[i]
  solvers.options["maxiters"] = 20
  solvers.options["show_progress"] = False
  result = solvers.qp( p, q, - g, h, a, b, "glpk" )
  r_w = w * 0.
  for i in xrange( len( w ) ):
    r_w[i] = result["x"][i]
  r_x = np.zeros( j + 1 )
  for i in xrange( j + 1 ):
    r_x[i] = result["x"][len( w ) + i]
  return r_w, r_x
