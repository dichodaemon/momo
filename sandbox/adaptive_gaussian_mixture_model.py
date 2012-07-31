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
  e1 = 2 * sqrt( s[0] )
  e2 = 2 * sqrt( s[1] )
  theta = np.arange( 0, 2 * pi, 0.01 )
  xx = e1 * np.cos( theta )
  yy = e2 * np.sin( theta )
  coord = np.vstack( [xx, yy] )
  coord = np.dot( U, coord )
  pl.plot( coord[0, :] + mu[0], coord[1, :] + mu[1], edge, linewidth = 2.0 )

class adaptive_gaussian_mixture_model( object ):
  def __init__( self, module ):
    self.module = module

  def learn( self, data, kmax = None ):
    kmin = 1
    if kmax == None:
      kmax = len( data ) / 50
    dim  = data[0].shape[0]
    N = dim + dim * ( dim + 1 ) / 2

    self.mu    = random.sample( data, kmax )
    self.sigma = []
    self.inv_sigma = []
    self.prior = np.array( [ 1.0 / kmax ] * kmax )

    weights = np.array( [1.0] * len( data ) )
    mu = self.module.mean( data, weights )
    cov = self.module.covariance( mu, data, weights )
    cov = 1. / kmax * cov
    #cov = 0.1 * np.average( np.diag( cov ) ) * np.eye( dim )
    #cov = 1. / ( kmax * N ) * np.average( np.diag( cov ) ) * np.eye( dim )
    for i in xrange( kmax ):
      self.sigma.append( 1 * cov )
      self.inv_sigma.append( np.linalg.inv( cov ) )

    self.optimize( kmin, kmax, N, data )


  def compute_expectation( self, prior, kmax, data, mu, sigma, inv_sigma ):
    k, n = [kmax, len( data )]
    result = np.zeros( (k, n) )
    for m in xrange( k ):
      if prior[m] > 0:
        self.update_expectation( result, m, data, mu, sigma, inv_sigma )
    return result

  def update_expectation( self, expectation, m, data, mu, sigma, inv_sigma ):
    n = data.shape[0]
    det  = np.linalg.det( sigma[m] )
    dim  = data.shape[1]
    norm = ( 2 * pi )**( -dim * 0.5 ) * det**( -0.5 )
    for i in xrange( n ):
      diff = self.module.difference( mu[m], data[i] )
      maha = np.dot( np.dot( diff, inv_sigma[m] ), np.transpose( diff ) )
      tmp  = norm * exp( -0.5 * maha )
      expectation[m, i] = tmp + 1E-300

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
    prior = self.prior * 1.0
    return mu, sigma, inv_sigma, prior


  def log_likelihood( self, actual_k, w_expectation ):
    k, n = w_expectation.shape
    result = 0
    for i in xrange( n ):
      z = np.sum( w_expectation[:, i] ) + 1E-300
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
    return tmp - self.log_likelihood( actual_k, w_expectation )

  def optimize( self, kmin, kmax, N, data ):
    epsilon = 1E-4
    expectation  = self.compute_expectation( self.prior, kmax, data, self.mu, self.sigma, self.inv_sigma )
    w_expectation = self.compute_w_expectation( expectation, self.prior )
    n_expectation = self.normalize_expectation( w_expectation )

    log_like = self.log_likelihood( kmax, w_expectation )
    cost     = self.cost( w_expectation, N, kmax, self.prior )
    cost_min = cost
    best     = self.clone_parameters()

    k = kmax
    n = data.shape[0]

    # Main Loop
    while k > kmin:
      old_log_like = -1E6
      count = 0
      while True:
        # Sequentially update all Gaussians
        for m in xrange( kmax ):
          if self.prior[m] > 0:
            w_expectation = self.compute_w_expectation( expectation, self.prior )
            n_expectation = self.normalize_expectation( w_expectation )

            weights = 1. * n_expectation[m, :] / np.sum( n_expectation[m, :] )
            self.mu[m] = self.module.mean( data, weights )
            self.sigma[m] = self.module.covariance( self.mu[m], data, weights )
            self.inv_sigma[m] = np.linalg.inv( self.sigma[m] )

            self.prior[m] = max( np.sum( n_expectation[m, :] ) - 0.5 * N, 0 ) / n
            self.prior = self.prior / np.sum( self.prior )

            old_k = k
            k = np.sum( ( self.prior > 0 ) )
            if old_k == k:
              self.update_expectation( expectation, m, data, self.mu, self.sigma, self.inv_sigma )
            else:
              print "k", k
              expectation[m, :] = 0.0
              w_expectation[m, :] = 0.0

        expectation  = self.compute_expectation( self.prior, kmax, data, self.mu, self.sigma, self.inv_sigma )
        w_expectation = self.compute_w_expectation( expectation, self.prior )
        n_expectation = self.normalize_expectation( w_expectation )
        if log_like > old_log_like or count > 10:
          old_log_like = log_like
          count = 0
        else:
          count += 1
        log_like = self.log_likelihood( k, w_expectation )
        cost = self.cost( w_expectation, N, k, self.prior )
        print k, cost
        if abs( ( log_like - old_log_like ) / old_log_like ) < epsilon:
          break

      k = np.sum( ( self.prior > 0 ) )

      if k > kmin:
        # Update best model
        if cost < cost_min:
          cost_min = cost
          best = self.clone_parameters()

        # Remove least relevant Gaussian
        min_idx = -1
        min_val = 1E20
        for m in xrange( kmax ):
          if self.prior[m] > 0 and self.prior[m] < min_val:
            min_val = self.prior[m]
            min_idx = m
        self.prior[min_idx] = 0
        k = np.sum( ( self.prior > 0 ) )
        print "New k=", k

        # Update expectation
        self.prior = self.prior / np.sum( self.prior )
        expectation  = self.compute_expectation( self.prior, kmax, data, self.mu, self.sigma, self.inv_sigma )
        w_expectation = self.compute_w_expectation( expectation, self.prior )
        log_like = self.log_likelihood( k, w_expectation )
        cost = self.cost( w_expectation, N, k, self.prior )
    # End of Main Loop
    self.mu = []
    self.sigma = []
    self.inv_sigma = []
    self.prior = []

    # Shrink Gaussian arrays
    mu, sigma, inv_sigma, prior = best
    for i in xrange( kmax ):
      if prior[i] > 0:
        self.mu.append( mu[i] )
        self.sigma.append( sigma[i] )
        self.inv_sigma.append( inv_sigma[i] )
        self.prior.append( prior[i] )
    self.k = len( self.prior )

    print "*"
    print cost_min, self.k
    print "*"



if __name__ == "__main__":
  import matplotlib.pylab as pl
  
  clusters = 3

  mu    = [ np.array( [0, m * 6. / clusters] ) for m in xrange( clusters )]
  sigma = [np.array( [ [ 2., 0.], [0.,  0.2] ] ) for m in xrange( clusters )]

  #mu[2] = mu[1] * 1
  #sigma[2] = np.array( [[ 0.5, 0], [0, 0.05] ] )

  data = []

  for m in xrange( len( mu ) ):
    for i in xrange( 300 ):
      noise = np.random.normal( size = mu[m].shape[0] )
      chol  = np.linalg.cholesky( sigma[m] )
      val   = mu[m] + np.dot( noise, chol )
      data.append( val )
  data = np.array( data )
  
  def begin_draw():
    pl.ion()
    pl.figure( 1, figsize = ( 20, 20 ), dpi = 50 )
    pl.clf()
    pl.axis( "scaled" )
    pl.xlim( -4, 4 )
    pl.ylim( -2, 6 )
    pl.plot( [d[0] for d in data], [d[1] for d in data], "b." )
    for m in xrange( clusters ):
      plot_gaussian( mu[m], sigma[m], "r" )

  def end_draw():
    pl.draw()
    #pl.show()

  #begin_draw()
  #end_draw()
  agmm = adaptive_gaussian_mixture_model( momo.features.default )
  agmm.learn( data )
  begin_draw()
  for m in xrange( agmm.k ):
    if agmm.prior[m] > 0.0:
      plot_gaussian( agmm.mu[m], agmm.sigma[m], str( agmm.prior[m] ) )
  end_draw()
  pl.ioff()
  pl.show()

