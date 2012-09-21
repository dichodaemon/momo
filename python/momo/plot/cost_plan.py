import pylab as pl
import numpy as np

def plot_cost( pos, angle, costs, path, vmax ):
  pl.subplot( 3, 3, pos )
  pl.axis( "scaled" )
  pl.xlim( 0, costs.shape[2] - 1 )
  pl.ylim( 0, costs.shape[1] - 1 )
  pl.imshow( costs[angle], pl.cm.jet, None, None, "none", vmin = 0, vmax = vmax )
  if path != None:
    pl.plot( 
      [ v[0] for v in path],
      [ v[1] for v in path],
      "m." 
    )
    pl.plot( 
      [path[-1][0]],
      [path[-1][1]],
      "m.", markersize = 20
    )

def cost_plan( plan, costs, path = None ):
  vmax = np.max( costs )
  plot_cost( 1, 3, costs, path, vmax )
  plot_cost( 2, 2, costs, path, vmax )
  plot_cost( 3, 1, costs, path, vmax )
  plot_cost( 4, 4, costs, path, vmax )
  plot_cost( 6, 0, costs, path, vmax )
  plot_cost( 7, 5, costs, path, vmax )
  plot_cost( 8, 6, costs, path, vmax )
  plot_cost( 9, 7, costs, path, vmax )

  pl.subplot( 3, 3, 5 )
  pl.axis( "scaled" )
  pl.xlim( 0, costs.shape[2] - 1 )
  pl.ylim( 0, costs.shape[1] - 1)
  pl.imshow( plan, pl.cm.jet, None, None, "none" )
  if path != None:
    pl.plot( 
      [ v[0] for v in path],
      [ v[1] for v in path],
      "m." 
    )
    pl.plot( 
      [path[-1][0]],
      [path[-1][1]],
      "m.", markersize = 20
    )
