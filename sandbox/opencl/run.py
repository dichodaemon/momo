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
    self.costs = loadProgram( self.context, "costs.cl" )
    self.dijkstra = loadProgram( self.context, "dijkstra.cl" )

    mf = cl.mem_flags
    fdirections = np.array( [
      [ 1.,  0.],
      [ 1.,  1.],
      [ 0.,  1.],
      [-1.,  1.],
      [-1.,  0.],
      [-1., -1.],
      [ 0., -1.],
      [ 1., -1.]
    ], dtype = np.float32 )
    idirections = fdirections.astype( np.int32 )
    angles = np.array( [
      [cos( pi / 8 ), sin( pi / 8 )],
      [cos( 3 * pi / 8 ), sin( 3 * pi / 8 )],
      [cos( 5 * pi / 8 ), sin( 5 * pi / 8 )],
      [cos( 7 * pi / 8 ), sin( 7 * pi / 8 )],
    ], dtype = np.float32 )
    speeds = np.array( [0., 0.02, 0.08], dtype = np.float32 )
    self.fdirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = fdirections )
    self.idirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = idirections )
    self.angle_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = angles )
    self.speed_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = speeds )

  def compute_weights( self, grid_width, grid_height, delta, speed, radius, theta, frame ):
    mf = cl.mem_flags

    costs = np.zeros( (8, grid_height, grid_width), dtype=np.float32 )

    theta_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = theta  )
    frame_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = frame  )
    cost_buffer  = cl.Buffer( self.context, mf.WRITE_ONLY, costs.nbytes )

    self.costs.computeWeights( 
      self.queue, costs.shape, None, 
      np.float32( speed ), np.float32( delta ), np.float32( radius ),
      np.int32( grid_width ), np.int32( grid_height ),
      np.int32( frame.shape[0] ), frame_buffer, 
      self.fdirection_buffer, self.angle_buffer, self.speed_buffer,      
      theta_buffer, cost_buffer 
    )

    cl.enqueue_read_buffer( self.queue, cost_buffer, costs ).wait()
    return costs

  def compute_dijkstra( self, grid_width, grid_height, costs, destination ):
    mf = cl.mem_flags

    floats = np.zeros( (8, grid_height, grid_width), dtype=np.float32 )
    ints   = np.zeros( (8, grid_height, grid_width), dtype=np.int32 )

    dest_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = destination )
    cost_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = costs )

    cummulated_buffer  = cl.Buffer( self.context, mf.READ_WRITE, floats.nbytes )
    tcummulated_buffer  = cl.Buffer( self.context, mf.READ_WRITE, floats.nbytes ) 
    parent_buffer  = cl.Buffer( self.context, mf.READ_WRITE, ints.nbytes )
    tparent_buffer  = cl.Buffer( self.context, mf.READ_WRITE, ints.nbytes )
    mask_buffer  = cl.Buffer( self.context, mf.READ_WRITE, ints.nbytes )

    e1 = self.dijkstra.initializeBuffers( 
      self.queue, costs.shape, None, 
      np.int32( grid_width ), np.int32( grid_height ),
      mask_buffer, cummulated_buffer, tcummulated_buffer, 
      parent_buffer, tparent_buffer, 
      dest_buffer
    )

    count = 0
    while True:
      self.dijkstra.dijkstraPass1( 
        self.queue, costs.shape, None, 
        np.int32( grid_width ), np.int32( grid_height ),
        self.idirection_buffer, cost_buffer,
        mask_buffer, cummulated_buffer, tcummulated_buffer, 
        tparent_buffer, wait_for = [e1] 
      )

      self.dijkstra.dijkstraPass2( 
        self.queue, costs.shape, None, 
        np.int32( grid_width ), np.int32( grid_height ),
        mask_buffer, cummulated_buffer, tcummulated_buffer, 
        parent_buffer, tparent_buffer
      )
      count += 1
      if count % 40 == 0:
        cl.enqueue_copy( self.queue, ints, mask_buffer )
        if not ints.any():
          break

    cl.enqueue_copy( self.queue, floats, cummulated_buffer )
    cl.enqueue_copy( self.queue, ints, parent_buffer )
    return floats, ints

def get_path( width, height, x, y, k, parents ):
  p1 = k * width * height + y * width + x
  x = []
  y = []
  while p1 != -1:
    i = p1 / ( height * width )
    j = ( p1 % ( height * width ) ) / width
    k = p1 % width
    p1 = parents[i, j, k]
    x.append( k * delta + 0.5 * delta )
    y.append( j * delta + 0.5 * delta )
  return x, y

if __name__ == "__main__":
  import time
  import pylab as pl
  data     = momo.read_data( "../../data/filtered/%s.txt" % sys.argv[1] )


  minx = min( data, key = lambda x: x[3] )[3]
  maxx = max( data, key = lambda x: x[3] )[3]
  miny = min( data, key = lambda y: y[4] )[4]
  maxy = max( data, key = lambda y: y[4] )[4]

  theta = np.random.rand( 18 ).astype( np.float32 )
  #theta = np.array( [0, 0, 0, 0, 1, 5, 5, 0, 3, 3, 0, 3, 3, 5, 0, 0, 0, 1], dtype = np.float32 )
  theta /= np.linalg.norm( theta )


  ts = sorted( theta )
  vmin = sum( ts[:2] )
  vmax = sum( ts[-2:] )

  solver = Solver()
  width  = 128
  height = 32
  delta  = 0.15

  pl.figure( 1, figsize = ( 30, 10 ), dpi = 75 )
  pl.ion()

  #costs = np.random.random( (8, height, width) )
  #costs = ( ( costs > 0.7 ) * 9.9 + 0.1 ).astype( np.float32 )

  for f in momo.frames( data ):
    frame = []
    for o in f:
      frame.append( [o[3] - minx, o[4] - miny, o[5], o[6]] )

    frame  = np.array( frame, dtype=np.float32 )

    t = time.time()

    costs = solver.compute_weights( width, height, delta, 0.04, 3, theta, frame )
    cummulated, parents = solver.compute_dijkstra( width, height, costs, np.array( [100, 24, 0], dtype = np.int32 ) )

    t = time.time() - t
    print "Compute FPS", 1 / t

    t = time.time()
    pl.clf()
    pl.subplot( 3, 3, 1 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[3], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 4 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[4], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 7 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[5], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 6 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[0], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 3 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[1], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 9 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[7], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 2 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[2], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 8 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( costs[6], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = vmin, vmax = vmax )
    pl.subplot( 3, 3, 5 )
    pl.axis( "scaled" )
    pl.xlim( 0, width * delta )
    pl.ylim( 0, height * delta )
    pl.imshow( cummulated[0], pl.cm.jet, None, None, "none", extent = (0, width * delta, 0, height * delta ), origin = "lower", vmin = 0, vmax = 80 )
    x, y = get_path( width, height, 0, 24, 0, parents )
    pl.plot( x, y, "m." )
    pl.draw()
    t = time.time() - t
    print vmin, vmax
    print "Draw FPS", 1 / t

