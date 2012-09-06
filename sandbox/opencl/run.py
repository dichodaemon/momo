#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os

BASE_DIR = os.path.abspath( os.path.join( os.path.dirname( __file__ ), "..", ".." ) )
path     = os.path.abspath( os.path.join( BASE_DIR, "python" ) )
sys.path.append( path )

import pyopencl as cl
import numpy as np
from math import *

import time
import momo
import momo.planning
import matplotlib.pylab as pl
import scipy.ndimage as ni

def loadProgram( context, filename ):
  f = open( filename, 'r' )
  fstr = "".join( f.readlines() )
  program = cl.Program( context, fstr ).build()
  return program

class Solver:
  def __init__(self):
    self.context = cl.create_some_context()
    self.queue = cl.CommandQueue( self.context )
    self.program = loadProgram( self.context, "computeFeature.cl" )

    mf = cl.mem_flags
    directions = np.array( [
      [ 1.,  0.],
      [ 1.,  1.],
      [ 0.,  1.],
      [-1.,  1.],
      [-1.,  0.],
      [-1., -1.],
      [ 0., -1.],
      [ 1., -1.]
    ], dtype = np.float32 )
    angles = np.array( [
      [cos( pi / 8 ), sin( pi / 8 )],
      [cos( 3 * pi / 8 ), sin( 3 * pi / 8 )],
      [cos( 5 * pi / 8 ), sin( 5 * pi / 8 )],
      [cos( 7 * pi / 8 ), sin( 7 * pi / 8 )],
    ], dtype = np.float32 )
    speeds = np.array( [0., 0.02, 0.08], dtype = np.float32 )
    self.direction_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = directions )
    self.angle_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = angles )
    self.speed_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = speeds )

  def compute_weights( self, grid_width, grid_height, delta, speed, theta, frame ):
    mf = cl.mem_flags
    costs = np.zeros( (8, grid_height, grid_width), dtype=np.float32 )
    theta_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = theta  )
    frame_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = frame  )
    cost_buffer  = cl.Buffer( self.context, mf.WRITE_ONLY, costs.nbytes )
    self.program.computeWeights( 
      self.queue, costs.shape, None, 
      np.float32( speed ), np.float32( delta ), 
      np.int32( grid_width ), np.int32( grid_height ),
      np.int32( frame.shape[0] ), frame_buffer, 
      self.direction_buffer, self.angle_buffer, self.speed_buffer,      
      theta_buffer, cost_buffer 
    )
    cl.enqueue_read_buffer( self.queue, cost_buffer, costs ).wait()
    return costs

if __name__ == "__main__":
  import time
  import pylab as pl
  data     = momo.read_data( "../../data/filtered/%s.txt" % sys.argv[1] )


  minx = min( data, key = lambda x: x[3] )[3]
  maxx = max( data, key = lambda x: x[3] )[3]
  miny = min( data, key = lambda y: y[4] )[4]
  maxy = max( data, key = lambda y: y[4] )[4]

  #theta = np.random.rand( 17 ).astype( np.float32 )
  theta = np.array( [1, 1, 1, 1, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1], dtype = np.float32 )
  theta = np.array( [1, 2, 3, 4, 1, 2, 3, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype = np.float32 )
  theta /= np.linalg.norm( theta )

  ts = sorted( theta )
  vmin = sum( ts[:2] )
  vmax = sum( ts[-2:] )

  solver = Solver()
  width  = 256
  height = 64
  delta  = 0.1

  pl.figure( 1, figsize = ( 10, 10 ), dpi = 75 )
  pl.ion()

  for f in momo.frames( data ):
    frame = []
    for o in f:
      frame.append( [o[3] - minx, o[4] - miny, o[5], o[6]] )
      #print o[5], 0.04 + o[5], 0.04 - o[5]

    frame  = np.array( frame, dtype=np.float32 )

    t = time.time()

    costs = solver.compute_weights( width, height, delta, 0.04, theta, frame )

    t = time.time() - t
    print "Compute FPS", 1 / t

    t = time.time()
    pl.clf()
    pl.imshow( costs[0], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.draw()
    t = time.time() - t
    print vmin, vmax
    print "Draw FPS", 1 / t

