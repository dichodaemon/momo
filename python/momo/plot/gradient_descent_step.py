import pylab as pl
import numpy as np
import momo

def gradient_descent_step( cummulated, costs, grid_path, error ):
  pl.figure( 1, figsize = (30, 5 ), dpi = 75 )
  pl.ion()
  pl.clf()
  momo.plot.cost_plan( np.sum( cummulated, 0 ), costs, grid_path )
  pl.subplots_adjust( left = 0.01, right = 0.99 )
  pl.text( 2, 2, "error: %f" % error, color = "w" )
  pl.draw()


