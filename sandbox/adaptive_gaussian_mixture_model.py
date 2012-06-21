#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os

BASE_DIR = os.path.abspath( os.path.join( os.path.dirname( __file__ ), ".." ) )
path     = os.path.abspath( os.path.join( BASE_DIR, "python" ) )
sys.path.append( path )

import momo
import numpy as np
from math import *
import random

def plot_gaussian( mu, sigma, edge = "k", face = "1.0" ):
  from matplotlib.patches import Ellipse
  U, s, Vh = np.linalg.svd( sigma )
  angle = atan2( U[1, 0], U[0, 0] )
  ellipse = Ellipse( 
    xy = mu, width = sqrt( s[0] ), height = sqrt( s[1] ),
    angle = angle, facecolor = face, edgecolor = edge 
  )
  ax = pl.gca()
  ax.add_patch( ellipse )

class adaptive_gaussian_mixture_model( object ):
  def __init__( self, module ):
    self.module = module

  def learn( self, data ):
    kmin = 1
    kmax = len( data ) / 3
    dim  = data[0].shape[0]
    N = dim + dim * ( dim + 1 ) / 2

    self.mu    = random.sample( data, kmax )
    self.sigma = []
    self.inv_sigma = []
    self.prior = np.array( [ 1.0 / kmax ] * kmax )
    for i in xrange( kmax ):
      self.sigma.append( np.eye( self.module.feature_size ) )
      self.inv_sigma.append( np.eye( self.module.feature_size ) )

    self.optimize( kmin, kmax, N, data )


  def compute_expectation( self, prior, kmax, data, mu, inv_sigma ):
    k, n = [kmax, len( data )]
    result = np.zeros( (k, n) )
    for m in xrange( k ):
      if prior[m] > 0.0:
        for i in xrange( n ):
          diff = self.module.difference( mu[m], data[i] )
          maha = np.dot( np.dot( diff, inv_sigma[m] ), np.transpose( diff ) )
          tmp  = exp( -0.5 * maha )
          result[m, i] = tmp
    return result

  def compute_w_expectation( self, expectation, prior ):
    k, n = expectation.shape
    result = np.zeros( (k, n) )
    for m in xrange( k ):
      result[m, :] = expectation[m, :] * prior[m]
    return result

  def normalize_expectation( self, w_expectation ):
    k, n = w_expectation.shape
    result = w_expectation * 1.0
    for i in xrange( n ):
      z = sum( result[:, i] )
      if z > 0.0:
        result[:, i] /= z
    return result

  def clone_parameters( self ):
    kmax = len( self.mu )
    mu = [self.mu[m] * 1.0 for m in xrange( kmax )]
    sigma     = [self.sigma[m] * 1.0 for m in xrange( kmax )]
    inv_sigma = [self.inv_sigma[m] * 1.0 for m in xrange( kmax )]
    prior = self.prior[m] * 1.0
    return mu, sigma, inv_sigma, prior

  def compute_prior( self, n_expectation, N ):
    k, n  = n_expectation.shape
    result = np.array( [0.] * k )
    for m in xrange( k ):
      result[m] = max( np.sum( n_expectation[m, :] ) - 0.5 * N, 0 ) / n
    result = result / np.sum( result )
    return result

  def log_likelihood( self, w_expectation ):
    k, n = w_expectation.shape
    result = 0
    for i in xrange( n ):
      z = np.sum( w_expectation[:, i] )
      if z > 0:
        result += log( z )
    return result

  def cost( self, w_expectation, N, actual_k, prior ):
    k, n = w_expectation.shape
    log_sum = 0
    for m in xrange( k ):
      if prior[m] > 0:
        log_sum += log( prior[m] )
    tmp  = 0.5 * N * log_sum
    tmp += (0.5 * N + 0.5 ) * actual_k * log( n )
    return tmp - self.log_likelihood( w_expectation )

  def optimize( self, kmin, kmax, N, data ):
    epsilon = 1E-5
    expectation  = self.compute_expectation( self.prior, kmax, data, self.mu, self.inv_sigma )
    w_expectation = self.compute_w_expectation( expectation, self.prior )
    n_expectation = self.normalize_expectation( w_expectation )

    log_like = self.log_likelihood( w_expectation )
    cost     = self.cost( w_expectation, N, kmax, self.prior )
    cost_min = cost
    best     = self.clone_parameters()

    k = kmax
    n = data.shape[0]

    # Main Loop
    while k >= kmin:
      while True:
        # Sequentially update all Gaussians
        for m in xrange( kmax ):
          if self.prior[m] > 0:
            w_expectation = self.compute_w_expectation( expectation, self.prior )
            n_expectation = self.normalize_expectation( w_expectation )

            weights = n_expectation[m, :] / np.sum( n_expectation[m, :] ) 
            self.mu[m] = self.module.mean( data, weights )
            self.sigma[m] = self.module.covariance( self.mu[m], data, weights )
            self.inv_sigma[m] = np.linalg.inv( self.sigma[m] )

            self.prior = self.compute_prior( n_expectation, N )

            if self.prior[m] <= 0.0:
              k = np.sum( ( self.prior > 0 ) )
            else:
              expectation  = self.compute_expectation( self.prior, kmax, data, self.mu, self.inv_sigma )

        expectation  = self.compute_expectation( self.prior, kmax, data, self.mu, self.inv_sigma )
        w_expectation = self.compute_w_expectation( expectation, self.prior )
        n_expectation = self.normalize_expectation( w_expectation )
        old_log_like = log_like
        log_like = self.log_likelihood( w_expectation )
        cost = self.cost( w_expectation, N, k, self.prior )

        if old_log_like - log_like < abs( epsilon * old_log_like ):
          break

      k = np.sum( ( self.prior > 0 ) )

      # Update best model
      if cost < cost_min:
        print k, cost, log_like
        cost_min = cost
        best = self.clone_parameters()
        begin_draw()
        for m in xrange( kmax ):
          if self.prior[m] > 0.0:
            plot_gaussian( self.mu[m], self.sigma[m], "r" )
        end_draw()

      # Remove least relevant Gaussian
      min_idx = -1
      min_val = 1E20
      for m in xrange( kmax ):
        if self.prior[m] > 0 and self.prior[m] < min_val:
          min_val = self.prior[m]
          min_idx = m
      if min_idx != -1:
        self.prior[min_idx] = 0
      k = np.sum( ( self.prior > 0 ) )
      k -= 1
    # End of Main Loop

if __name__ == "__main__":
  import matplotlib.pylab as pl

  mu    = [ np.array( [0, m * 5] ) for m in xrange( 3 )]
  sigma = [np.array( [ [ 10., 0.], [0.,  5.] ] ) for m in xrange( 3 )]

  data = []

  for m in xrange( 3 ):
    for i in xrange( 500 ):
      data.append( np.random.multivariate_normal( mu[m], sigma[m] ) )
  data = np.array( data )
  
  def begin_draw():
    #pl.ion()
    pl.figure( 1 )
    pl.clf()
    pl.axis( "scaled" )
    pl.xlim( -20, 20 )
    pl.ylim( -15, 25 )
    for m in xrange( 3 ):
      plot_gaussian( mu[m], sigma[m] )
    pl.plot( [d[0] for d in data], [d[1] for d in data], "b." )

  def end_draw():
    #pl.draw()
    pl.show()

  begin_draw()
  end_draw()
  agmm = adaptive_gaussian_mixture_model( momo.features.default )
  agmm.learn( data )

