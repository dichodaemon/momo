import pylab as pl
import numpy as np


def plot_features( convert, frame, features ):
  nfeatures = features.shape[3]
  nangles   = features.shape[0]

  pl.figure( 2, figsize = ( ( nfeatures + 1 ) * 5 , ( nangles + 1 ) * 2 ), dpi = 50 )
  pl.ion()
  pl.clf()

  vmax = np.max( features )

  for f_idx in xrange( nfeatures ):
    for a_idx in xrange( nangles ):
      pl.subplot( nangles, nfeatures, a_idx * nfeatures + f_idx + 1 )
      pl.xlim( convert.x, convert.x2 )
      pl.ylim( convert.y, convert.y2 )
      pl.imshow( features[a_idx, :, :, f_idx], pl.cm.Greys, None, None, "none", vmin = 0, vmax = vmax, extent = ( convert.x, convert.x2, convert.y2, convert.y ) )
      pl.plot( frame[:, 0], frame[:, 1], 'b.', markersize = 1 )


  pl.subplots_adjust( left = 0.01, right = 0.99 )
  pl.draw()


