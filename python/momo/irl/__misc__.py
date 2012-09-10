import momo
import numpy as np

def preprocess_data( convert, data ):
  frame_data = {}
  for frame in momo.frames( data ):
    tmp = []
    for o_frame, o_time, o_id, o_x, o_y, o_dx, o_dy in frame:
      tmp.append( [o_id, np.array( [o_x, o_y, o_dx, o_dy] )] )

    for i in xrange( len( frame ) ):
      o_id, o = tmp.pop( 0 )
      if not o_id in frame_data:
        frame_data[o_id] = {
          "states": [], 
          "frames": []
        }
      frame_data[o_id]["states"].append( o )
      frame_data[o_id]["frames"].append( [f[1] for f in tmp] )
      tmp.append( [o_id, o] )

  # Rasterize to grid
  for o_id, frame in frame_data.items():
    old_grid_s = None
    states = []
    frames = []
    for index in xrange( len( frame["states"] ) ):
      state = frame["states"][index]
      angle = momo.angle.as_angle( np.array( state[2:] ) )
      grid_s = convert.from_world( np.array( [state[0], state[1], angle] ) )
        if old_grid_s == None or old_grid_s != grid_s:
          old_grid_s = grid_s
          states.append( frame["states"][index] )
          frames.append( frame["frames"][index] )
    frame["states"] = states
    frame["frames"] = frames
  return frame_data

