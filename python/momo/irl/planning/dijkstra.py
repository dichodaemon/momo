import momo

class dijkstra( object ):
  def __init__( self, convert, compute_costs ):
    self.convert = convert
    self.compute_costs = compute_costs
    self.planner = momo.planning.dijkstra()

  def __call__( self, start, goal, velocity, frame, theta ):
    current = self.convert.from_world2( start )
    goal = self.convert.from_world2( goal )

    costs = self.compute_costs( velocity, theta, frame )
    cummulated, parents = self.planner( costs, goal )
    path = self.planner.get_path( parents, current )

    result = []
    for p in path:
      result.append( self.convert.to_world2( p, velocity ) )
    return result, cummulated, costs
