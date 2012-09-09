import pyopencl as cl
import numpy as np
from math import *

class Program( object ):
  def __init__( self ):
    self.context = cl.create_some_context()
    self.queue =cl.CommandQueue( self.context )

  def loadProgram( self, filename ):
    f = open( filename, 'r' )
    fstr = "".join( f.readlines() )
    program = cl.Program( self.context, fstr ).build()
    return program

