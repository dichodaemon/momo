import pyopencl as cl
import numpy as np
import momo
from math import *

class Program( object ):
  def __init__( self ):
    self.context = cl.create_some_context()
    self.queue =cl.CommandQueue( self.context )

  def loadProgram( self, filename ):
    f = open( filename, 'r' )
    fstr = "".join( f.readlines() )
    program = cl.Program( self.context, fstr ).build( "-I %s/opencl/" % momo.BASE_DIR )
    return program

