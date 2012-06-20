import momo
import numpy as np
from math import *
import random

class growing_gaussian_mixture( object ):
  def __init__( self, module, p1, p2 ):
    self.kmin = 1
    self.kmax = len( p1 ) / 3
    self.module = module
    self.N = module.feature_size  + module.feature_size * ( module.feature_size + 1 ) / 2.0

    features = momo.compute_features( module, p1, p2 )
    
    self.mu    = random.sample( features, self.kmax )
    self.sigma = []
    self.inv_sigma = []
    self.prior = np.array( [ 1.0 / self.kmax ] * self.kmax )
    for i in xrange( self.kmax ):
      self.sigma.append( np.eye( module.feature_size ) )
      self.inv_sigma.append( np.eye( module.feature_size ) )

    self.optimize( features )

  def compute_u( self, k, features, u = None, onem = None ):
    n = len( features )
    if u == None:
      u = np.zeros( ( k, n ) )
    for i in xrange( n ):
      for m in xrange( k ):
        if m == onem or onem ==  None:
          diff = self.module.difference( self.mu[m], features[i] )
          maha = np.dot( np.dot( diff, self.inv_sigma[m] ), np.transpose( diff ) )
          tmp  = exp( -0.5 * maha )
          u[m, i] = tmp
    return u

  def compute_w( self, u, prior ):
    k, n = u.shape
    w  = np.zeros( u.shape )
    wn = np.zeros( u.shape )
    for i in xrange( n ):
      w[:, i] = u[:, i] * prior
      wn[:, i] = w[:, i] / np.sum( w[:, i] )
    return w, wn

  def compute_prior( self, w ):
    k, n = w.shape
    prior = np.array( [0.0] * k )
    z = 0.0
    for m in xrange( k ):
      tmp = 0
      for i in xrange( n ):
        tmp += w[m, i]
      tmp -= 0.5 * self.N
      if tmp < 0:
        tmp = 0
      prior[m] = k
    prior = prior / np.sum( prior )
    return prior

  def clone_parameters( self ):
    mu = [self.mu[m] * 1.0 for m in xrange( self.kmax )]
    sigma     = [self.sigma[m] * 1.0 for m in xrange( self.kmax )]
    inv_sigma = [self.inv_sigma[m] * 1.0 for m in xrange( self.kmax )]
    prior = self.prior[m] * 1.0
    return mu, sigma, inv_sigma, prior


  def log_likelihood( self, u ):
    k, n = u.shape
    result = 0.0
    for i in xrange( n ):
      tmp = 0
      for m in xrange( k ):
        tmp += u[m, i]
      if tmp > 0:
        result += np.log( tmp )
    return result

  def criterion( self, u ):
    k, n = u.shape
    first = 0.0
    for m in xrange( k ):
      if self.prior[m] > 0:
        first += log( self.prior[m] )
    first = self.N / 2.0 * first\
            + ( self.N / 2.0 + 0.5 ) * self.k * log( n  )
    #for m in xrange( self.kmax ):
      #if self.prior[m] > 0:
        #first += log( n * self.prior[m] / 12 )
    #first = self.N / 2.0 * first\
            #+ self.k / 2.0 * log( n / 12 )\
            #+ ( self.k * self.N + self.k ) / 2.0
    #second = 0.0
    #for i in xrange( n ):
      #tmp = 0
      #for m in xrange( self.kmax ):
        #tmp += self.prior[m] * u[m][i]
      #second += log( tmp )
    return first - self.log_likelihood( u )

  def optimize( self, features ):
    self.k = self.kmax
    lmin = 1E20
    epsilon = 1E-5
    u = self.compute_u( self.kmax, features )
    w, wn = self.compute_w( u, self.prior )
    l = self.criterion( w )
    ll = self.log_likelihood( w )
    best = self.clone_parameters()
    while self.k >= self.kmin:
      while True:
        for m in xrange( self.kmax ):
          w, wn = self.compute_w( u, self.prior )
          old_prior = self.prior[m]
          #self.prior = self.compute_prior( w )
          if self.prior[m] > 0:
            weights = wn[m, :]
            self.mu[m] = self.module.mean( features, weights )
            #print self.sigma[m]
            self.sigma[m] = self.module.covariance( self.mu[m], features, weights )
            #print self.sigma[m]
            self.inv_sigma[m] = np.linalg.inv( self.sigma[m] )
            self.compute_u( self.kmax, features, u, m )
          else:
            if old_prior > 0:
              self.k -= 1
        old_ll = ll
        l = self.criterion( w )
        ll = self.log_likelihood( w )
        print self.k, l, ll, old_ll, epsilon * old_ll
        if old_ll - ll < abs( epsilon * old_ll ):
          break
      if l < lmin:
        print self.k, l
        lmin = l
        best = self.clone_parameters()

      min_idx = -1
      min_val = 1E20
      for m in xrange( self.kmax ):
        if self.prior[m] > 0 and self.prior[m] < min_val:
          min_val = self.prior[m]
          min_idx = m
      if min_idx != -1:
        self.prior[min_idx] = 0
        self.k -= 1
      ll = self.log_likelihood( u )
    self.mu = []
    self.sigma = []
    self.inv_sigma = []
    self.prior = []
    mu, sigma, inv_sigma, prior = best
    self.k = 0
    for m in xrange( self.kmax ):
      if prior[m] > 0:
        self.k += 1
        self.mu.append( mu[m] )
        self.sigma.append( sigma[m] )
        self.inv_sigma.append( inv_sigma[m] )
        self.prior.append( prior[m] )
    print "k", self.k


  def __call__( self, v1, v2 ):
    value = self.module.compute( v1, v2 )
    cost  = 0.0
    for ki in xrange( self.k ):
      diff  = self.module.difference( self.mu[ki], value )
      cost += (self.prior[ki] * np.dot( np.dot( diff, self.inv_sigma[ki] ), np.transpose( diff ) ) )**0.5 
    return cost



function result = mvnrnd( mu, Sigma, K )
  if nargin == 3
    mu = repmat( mu, K, 1 );
  end
  [n, d] = size( mu );
  try
    U = chol( Sigma );
  catch
    [e, Lambda] = eig( Sigma );
    U = sqrt( Lambda ) * E';
  end
  result = randn( n, d ) * U + mu;
end

