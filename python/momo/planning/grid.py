import numpy as np
from math import *

angles     = [0, pi / 4, pi / 2, 3 * pi / 4, pi, -3 * pi / 4, -pi / 2, -pi / 4]
directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
directions = [np.array( d ) for d in directions]

class grid( object ):
  def __init__( self, x, y, width, height, delta ):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.delta = delta
    self.grid_width = int( ceil( width / delta ) )
    self.grid_height = int( ceil( height / delta ) )

    self.width = self.delta * self.grid_width
    self.height = self.delta * self.grid_height
    self.grid = np.zeros( ( self.grid_width, self.grid_height, 8 ) )

  def __getitem__( self, coords ):
    i, j, k = coords
    return self.grid[i, j, k]

  def __setitem__( self, coords, value ):
    i, j, k = coords
    self.grid[i, j, k] = value

  def __iter__( self ):
    for k in xrange( 8 ):
      y = self.y + self.height - self.delta / 2.
      for j in xrange( self.grid_height ):
        x = self.x + self.delta / 2.
        for i in xrange( self.grid_width ):
          yield i, j, k, x, y, angles[k], self.grid[i, j, k]
          x += self.delta
        y -= self.delta

  def neighbors( self, i, j, k ):
    for td in xrange( k - 1, k + 2 ):
      d = td % 8
      ni = i + directions[d][0]
      nj = j - directions[d][1]
      nk = d
      if ni < self.grid_width and\
         ni >= 0 and\
         nj < self.grid_height and\
         nj >= 0:
        yield ni, nj, nk
  
  def from_world( self, x, y, angle ):
    while angle >  pi:
      angle -= 2 * pi
    while angle < - pi:
      angle += 2 * pi
    dist = 2 * pi
    k = 0
    for i in xrange( len( angles ) ):
      a = angles[i]
      d = abs( a - angle )
      if d < dist:
        k = i
        dist = d
    return int( floor( ( x - self.x ) / self.delta ) ),\
           int( floor( ( self.y + self.height - y ) / self.delta ) ), k

  def to_world( self, i, j, k ):
    return self.x + self.delta * ( i + 0.5 ), self.y + self.height - self.delta * ( j + 0.5 ), angles[k]


if __name__ == "__main__":
  g = grid( 1, 2, 3, 4, 1 )
  print g.to_world( 0, 0, 1 )
  print g.to_world( 2, 3, 1 )
  print g.from_world( 3.5, 5.5, 0 )
  print g.from_world( 1.5, 2.5, 0 )
  #for i, j, k, x, y, angle, value in g:
    #print i, j, k
    #for ni, nj, nk in g.neighbors( i, j, k ):
      #print "\t", ni, nj, nk

