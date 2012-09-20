import momo

class forward_backward( object ):
  def __init__( self, convert, compute_costs ):
    self.convert = convert
    self.compute_costs = compute_costs
    self.planner = momo.planning.forward_backward()

  def __call__( self, start, goal, velocity, frame, theta ):
    start = self.convert.from_world2( start )
    goal = self.convert.from_world2( goal )

    momo.tick( "FB Costs" )
    costs = self.compute_costs( velocity, theta, frame )
    momo.tack( "FB Costs" )
    #momo.tick( "FB Forward" )
    forward = self.planner( costs, start, 1 )
    #momo.tack( "FB Forward" )
    #momo.tick( "FB Backward" )
    backward = self.planner( costs, goal, -1 )
    #momo.tack( "FB Backward" )
    return forward, backward, costs

