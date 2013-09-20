#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os

BASE_DIR = os.path.abspath( os.path.join( os.path.dirname( __file__ ), "..", "..", ".." ) )
path     = os.path.abspath( os.path.join( BASE_DIR, "python" ) )
sys.path.append( path )

import pyopencl as cl
import numpy as np
import pylab as pl
from math import *
import momo
from __common__ import *
import time

class forward_backward( momo.opencl.Program ):
  def __init__( self ):
    momo.opencl.Program.__init__( self )
    self.flow = self.loadProgram( momo.BASE_DIR + "/opencl/forwardBackward.cl" )

    mf = cl.mem_flags

    self.idirection_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = DIRECTIONS )

  def __call__( self, costs, start, goal ):
    if ( costs < 0 ).any():
      raise runtime_error( "The cost matrix cannot have negative values" )
    mf = cl.mem_flags

    width = costs.shape[2]
    height = costs.shape[1]


    forward = np.zeros( costs.shape, dtype=np.float64 )
    f_masks = np.zeros( costs.shape, dtype=np.int32 )
    f_masks[tuple( reversed( start.tolist() ) )] = 1
    forward[tuple( reversed( start.tolist() ) )] = 1

    backward = np.zeros( costs.shape, dtype=np.float64 )
    b_masks = np.zeros( costs.shape, dtype=np.int32 )
    b_masks[tuple( reversed( goal.tolist() ) )] = 1
    backward[tuple( reversed( goal.tolist() ) )] = 1

    cost_buffer  = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = costs.astype( np.float64 ) )

    f_mask_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = f_masks )
    f1_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = forward )
    f2_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = forward )

    b_mask_buffer = cl.Buffer( self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = b_masks )
    b1_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = backward )
    b2_buffer  = cl.Buffer( self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = backward )

    wait = []

    for i in xrange( sum( costs.shape ) ):
      momo.tick( "first + second" )
      momo.tick( "first" )
      e1 = self.flow.forwardPass( 
        self.queue, costs.shape, None, 
        np.int32( width ), np.int32( height ),
        self.idirection_buffer, cost_buffer,
        f_mask_buffer,  
        b_mask_buffer,  
        f1_buffer, f2_buffer, 
        b1_buffer, b2_buffer, 
        wait_for = wait
      )
      wait = [e1]
      momo.tack( "first" )
      momo.tick( "second" )
      e2 = self.flow.updatePass( 
        self.queue, costs.shape, None, 
        np.int32( width ), np.int32( height ),
        f1_buffer, f2_buffer, 
        b1_buffer, b2_buffer, 
        wait_for = wait
      )
      wait = [e2]
      momo.tack( "second" )
      momo.tack( "first + second" )
    momo.tick( "wait" )
    e2.wait()
    momo.tack( "wait" )
    momo.tick( "copy" )
    cl.enqueue_copy( self.queue, forward, f1_buffer )
    cl.enqueue_copy( self.queue, backward, b1_buffer )
    momo.tack( "copy" )
    return forward, backward


if __name__ == "__main__":
  costs = np.ones( (8, 7, 8 ) )
  for i in xrange( 4 ):
    costs[i] *= ( i + 1 ) * 10
    if i != 0:
      costs[8 - i] *= ( i + 1 ) * 10

  start = np.array( [1, 3, 0], dtype = np.int32 )
  end = np.array( [6, 3, 0], dtype = np.int32 )

  fb = forward_backward()

  f, b = fb( costs, start, end )
  print b[0, 3, 1], f[6, 3, 0]

