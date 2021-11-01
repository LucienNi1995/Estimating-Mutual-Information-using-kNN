# Keith Briggs & Haoran Ni 2018-06-27
# python3 Methods_Collection_00.py

'''Collect all the methods in this module'''

import numpy as np
from scipy.spatial import cKDTree,KDTree
from scipy.special import gamma,digamma
from math import log,sqrt
from scipy.special import psi

def KL_h(samples, n,d, k, norm):
  # apply p-norm to compute log_c_d 
  if norm == np.inf: # max norm 
    log_c_d = 0
  elif norm == 2: # euclidean norm
    log_c_d = (d/2.) * np.log(np.pi) -np.log(gamma(d/2. +1))
  elif norm == 1:
    raise NotImplementedError
  else:
    raise NotImplementedError("Variable 'norm' either 1, 2 or np.inf")
  kdtree = cKDTree(samples)
  distances, idx = kdtree.query(samples, k + 1, eps=0, p=norm)
  # 2:radius -> diameter
  if norm==2:
    return -digamma(k) + digamma(n) + log_c_d + (d / n) * np.sum(np.log(distances[:, -1]))
  else:
    return -digamma(k) + digamma(n) + log_c_d + (d / n) * np.sum(np.log(2*distances[:, -1]))

def threeKL_MI(samples,k,norm):
  n,d=samples.shape
  return KL_h(samples[:,:d//2],n,d//2,  k=k, norm=norm)+KL_h(samples[:,d//2:],n,d//2,  k=k, norm=norm)-KL_h(samples,n,d, k=k, norm=norm)

def KSG1_MI(samples,k,norm):
  # store data points in kd-trees for efficient nearest neighbour computations
  n,d=samples.shape
  x_tree  = cKDTree(samples[:,:d//2])
  y_tree  = cKDTree(samples[:,d//2:])
  xy_tree = cKDTree(samples)
  # kth nearest neighbour distances
  # query with k=k+1 to return the nearest neighbour, not counting the data point itself
  dist, idx = xy_tree.query(samples, k=k+1, p=norm)
  epsilon = dist[:, -1]-(1e-15)
  # for each point, count the number of neighbours
  # whose distance in the x-subspace is strictly < epsilon
  # repeat for the y subspace
  nx = np.empty(n, dtype=np.int)
  ny = np.empty(n, dtype=np.int)
  for ii in range(n):
    nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon[ii], p=norm)) - 1
    ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon[ii], p=norm)) - 1
  return digamma(k) - np.mean(digamma(nx+1) + digamma(ny+1)) + digamma(n)  # Alg.1

def KSG2_MI(samples,k,norm):
  # store data points in kd-trees for efficient nearest neighbour computations
  n,d=samples.shape
  x_tree  = cKDTree(samples[:,:d//2])
  y_tree  = cKDTree(samples[:,d//2:])
  xy_tree = cKDTree(samples)
  # kth nearest neighbour distances
  # query with k=k+1 to return the nearest neighbour, not counting the data point itself
  dist, idx = xy_tree.query(samples, k=k+1, p=norm)
  epsilon1=np.linalg.norm((samples[:,:d//2]-samples[idx[:,k],:d//2]),norm,axis=1)
  epsilon2=np.linalg.norm((samples[:,d//2:]-samples[idx[:,k],d//2:]),norm,axis=1)
  # for each point, count the number of neighbours
  # whose distance in the x-subspace is strictly < epsilon
  # repeat for the y subspace
  nx = np.empty(n, dtype=np.int)
  ny = np.empty(n, dtype=np.int)
  for ii in range(n):
    nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon1[ii], p=norm)) - 1
    ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon2[ii], p=norm)) - 1
  return digamma(k) -1.0 -np.mean(digamma(nx) + digamma(ny)) + digamma(n) # Alg.2


def BIKSG_MI(samples,k,norm): #waiting for debugging
  n,d=samples.shape
  LogVol=np.log(gamma(d/2+1)/(gamma(d/4+1)*gamma(d/4+1)))
  x_tree  = cKDTree(samples[:,:d//2])
  y_tree  = cKDTree(samples[:,d//2:])
  xy_tree = cKDTree(samples)
  dist, idx = xy_tree.query(samples, k=k+1, p=norm)
  epsilon = dist[:, -1]
  nx = np.empty(n, dtype=np.int)
  ny = np.empty(n, dtype=np.int)
  for ii in range(n):
    nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon[ii], p=norm)) - 1
    ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon[ii], p=norm)) - 1
  return digamma(k)+np.log(n)+LogVol-np.mean(np.log(nx)+np.log(ny))

def Gknn_h(samples,n,d,k):
  x_tree=cKDTree(samples)
  dist,idx=x_tree.query(samples,k=k+1,p=2)
  term4=0.0
  nx=np.empty(n,dtype=np.int)
  #if d==1: 
   # v=np.empty(n)
    #for ii in range(n):
     # samplex=samples[idx[ii,:],:]
     # u, s, vh = np.linalg.svd(samplex-np.mean(samplex,axis=0), full_matrices=True)    
     # s=s/s[0]*dist[ii, -1]
     # samplex=(samples[idx[ii,:],:]-samples[ii,:])
     # ellip=np.einsum('ij,jm,km->ik',np.diag(1.0/s),vh,samplex)
      #ellip=np.dot(np.diag(1.0/s),np.dot(vh,samplex.T))
     # ellip=np.einsum('ji,ji->i',ellip,ellip)
      #ellip=np.diag(np.dot(ellip.T,ellip))
     # nx[ii]=len(ellip[ellip<=1])
     # v[ii]=2*s/vh
    #return -np.mean(np.log(nx/n/v))
  #else:
  for ii in range(n):
    samplex=samples[idx[ii,:],:]
    u, s, vh = np.linalg.svd(samplex-np.mean(samplex,axis=0), full_matrices=True)
    s=s/s[0]*dist[ii, -1]
    term4+=np.sum(np.log(s))
    samplex=(samples[idx[ii,:],:]-samples[ii,:])
    ellip=np.einsum('ij,jm,km->ik',np.diag(1.0/s),vh,samplex)
      #ellip=np.dot(np.diag(1.0/s),np.dot(vh,samplex.T))
    ellip=np.einsum('ji,ji->i',ellip,ellip)
      #ellip=np.diag(np.dot(ellip.T,ellip))
    nx[ii]=len(ellip[ellip<=1])   
  return np.log(n)+np.log(np.pi**(d/2.0)/gamma(d/2.0+1.0))-np.mean(np.log(nx))+term4/n#+np.mean(0.5/nx)#+np.mean(np.log(nx)-digamma(nx))

def Gknn_MI(samples,k):
  n,d=samples.shape
  return Gknn_h(samples[:,:d//2],n,d//2,  k=k)+Gknn_h(samples[:,d//2:],n,d//2,  k=k)-Gknn_h(samples,n,d, k=k)

def Gknnnew_h(samples,n,d,k):
  x_tree=cKDTree(samples)
  dist,idx=x_tree.query(samples,k=k+1,p=2)
  term4=0.0
  #pp=0.0
  nx=np.empty(n,dtype=np.int)
  #if d==1: 
   # v=np.empty(n)
    #for ii in range(n):
     # samplex=samples[idx[ii,:],:]
     # u, s, vh = np.linalg.svd(samplex-np.mean(samplex,axis=0), full_matrices=True)    
     # s=s/s[0]*dist[ii, -1]
     # samplex=(samples[idx[ii,:],:]-samples[ii,:])
     # ellip=np.einsum('ij,jm,km->ik',np.diag(1.0/s),vh,samplex)
      #ellip=np.dot(np.diag(1.0/s),np.dot(vh,samplex.T))
     # ellip=np.einsum('ji,ji->i',ellip,ellip)
      #ellip=np.diag(np.dot(ellip.T,ellip))
     # nx[ii]=len(ellip[ellip<=1])
     # v[ii]=2*s/vh
    #return -np.mean(np.log(nx/n/v))
  #else:
  for ii in range(n):
    samplex=samples[idx[ii,:],:]
    u, s, vh = np.linalg.svd(samplex-np.mean(samplex,axis=0), full_matrices=True)
    #s1=s/s[0]
    #print(np.multiply.reduce(s/s[0]))
    s=s/s[0]*dist[ii, -1]
    term4+=np.sum(np.log(s))
    samplex=(samples[idx[ii,:],:]-samples[ii,:])
    ellip=np.einsum('ij,jm,km->ik',np.diag(1.0/s),vh,samplex)
      #ellip=np.dot(np.diag(1.0/s),np.dot(vh,samplex.T))
    ellip=np.einsum('ji,ji->i',ellip,ellip)
      #ellip=np.diag(np.dot(ellip.T,ellip))
    nx[ii]=len(ellip[ellip<=1])-1
    if nx[ii]==0:
      term4-=0.0     
    else:
      term4-=np.log(nx[ii])   
    #pp+=(log(nx[ii])-digamma(nx[ii]))  
  return np.log(n)+np.log(np.pi**(d/2.0)/gamma(d/2.0+1.0))+term4/n#+pp/n#+np.mean(0.5/nx)#+np.mean(np.log(nx)-digamma(nx))

def Gknnnew_MI(samples,k):
  n,d=samples.shape
  return Gknnnew_h(samples[:,:d//2],n,d//2,  k=k)+Gknnnew_h(samples[:,d//2:],n,d//2,  k=k)-Gknnnew_h(samples,n,d, k=k)

def disc_cont_MI(samples,possiblex,k):
  n,d=samples.shape
  nnx=possiblex.shape[0]
  x_tree=cKDTree(samples[:,d//2:])
  term1=np.empty(nnx,dtype='float')
  term2=np.empty(nnx,dtype='float')
  for jj in range(nnx):
    samplex=samples[np.all((samples[:,:d//2]==possiblex[jj,:]),axis=1)]
    #& (samples[:,1]==possiblex[jj,1])]
    Nx=samplex.shape[0]
    xi_tree  = cKDTree(samplex[:,d//2:])
    dist, idx = xi_tree.query(samplex[:,d//2:], k=k+1, p=2) 
    m=np.empty(Nx,dtype=np.int)
    for ii in range(Nx):
      m[ii] = len(x_tree.query_ball_point(xi_tree.data[ii], r=dist[ii,-1], p=2))-1
    term1[jj]=digamma(Nx)*Nx
    term2[jj]=np.sum(digamma(m))
  return digamma(n)+digamma(k)-np.sum(term1)/n-np.sum(term2)/n

def Ross_discrete_continuous_MI_estimate(samples,k=1):
  # by Keith!
  I,y,n=0.0,0.0,0
  kdt=cKDTree(np.vstack([samples[x] for x in samples])) 
  for j,x in enumerate(samples):
    kdtx=cKDTree(samples[x])
    y+=digamma(len(samples[x]))*len(samples[x])
    rx,_=kdtx.query(samples[x],k=k+1,p=2)
    for i,xi in enumerate(samples[x]):
 # distance to kth neighbour
      mi=len(kdt.query_ball_point(xi,r=rx[i,-1],p=2))-1
      Ii=digamma(mi)
      I+=(Ii-I)/(1.0+n)
      n+=1
  return digamma(n)+digamma(k)-I-y/n

def dc_Gknn_h(samples,n,d,k):
  x_tree=cKDTree(samples)
  dist,idx=x_tree.query(samples,k=k+1,p=2)
  term4=0.0
  nx=np.empty(n,dtype=np.int)
  for ii in range(n):
    samplex=samples[idx[ii,:],:]
    u, s, vh = np.linalg.svd(samplex-np.mean(samplex,axis=0), full_matrices=True)
    s=s/s[0]*dist[ii, -1]
    term4+=np.sum(np.log(s))
    samplex=(samples[idx[ii,:],:]-samples[ii,:])
    ellip=np.einsum('ij,jm,km->ik',np.diag(1.0/s),vh,samplex)
      #ellip=np.dot(np.diag(1.0/s),np.dot(vh,samplex.T))
    ellip=np.einsum('ji,ji->i',ellip,ellip)
      #ellip=np.diag(np.dot(ellip.T,ellip))
    nx[ii]=len(ellip[ellip<=1])   
  return term4/n-np.mean(np.log(nx))


def dc_MI_gknn(samples,k=20):
  samplestotal=np.vstack([samples[x] for x in samples])
  n,d=samplestotal.shape
  dcmi=dc_Gknn_h(samplestotal,n,d,k) 
  term1=log(n)
  for j,x in enumerate(samples):
    nx=samples[x].shape[0]
    p=nx/n
    dcmi-=p*dc_Gknn_h(samples[x],nx,d,k)
    term1-=p*log(nx)
  return term1+dcmi

def dc_MI_gknn2(samples,possiblex,k=20):
  n,d=samples.shape
  dcmi=dc_Gknn_h(samples[:,d//2:],n,d//2,k) 
  term1=log(n)
  nnx=possiblex.shape[0]
  for jj in range(nnx):
    samplex=samples[np.all((samples[:,:d//2]==possiblex[jj,:]),axis=1)]
    nx=samplex.shape[0]
    p=nx/n
    dcmi-=p*dc_Gknn_h(samplex[:,d//2:],nx,d//2,k)
    term1-=p*log(nx)
  return term1+dcmi

def kldirect(samples,k=20):
  samplestotal=np.vstack([samples[x] for x in samples])
  n,d=samplestotal.shape
  dcmi=KL_h(samplestotal,n,d,k,norm=np.inf) 
  for j,x in enumerate(samples):
    nx=samples[x].shape[0]
    p=nx/n
    dcmi-=p*KL_h(samples[x],nx,d,k,norm=np.inf) 
  return dcmi

def kldirect2(samples,possiblex,k=20):
  n,d=samples.shape
  dcmi=KL_h(samples[:,d//2:],n,d//2,k,norm=np.inf) 
  nnx=possiblex.shape[0]
  for jj in range(nnx):
    samplex=samples[np.all((samples[:,:d//2]==possiblex[jj,:]),axis=1)]
    nx=samplex.shape[0]
    p=nx/n
    if nx==0:
      dcmi=0.0
    else:
      dcmi-=p*KL_h(samplex[:,d//2:],nx,d//2,k,norm=np.inf)
  return dcmi

def multiKLnew(samples,possiblex,norm,k=20):
  n,d=samples.shape
  kdtree2 = cKDTree(samples[:,d//2:])
  if norm == np.inf: # max norm 
    log_c_d = 0
  elif norm == 2: # euclidean norm
    log_c_d = (d/2.) * np.log(np.pi) -np.log(gamma(d/2. +1))
  elif norm == 1:
    raise NotImplementedError
  else:
    raise NotImplementedError("Variable 'norm' either 1, 2 or np.inf")
  dcmi2=0.0
  dcmi=0.0
  nnx=possiblex.shape[0]
  for jj in range(nnx):
    samplex=samples[np.all((samples[:,:d//2]==possiblex[jj,:]),axis=1)]
    nx=samplex.shape[0]
    p=nx/n
    if nx==0:
      dcmi=0.0
    else:
      kdtree = cKDTree(samplex[:,d//2:])
      distances, idx = kdtree.query(samplex[:,d//2:], k + 1, eps=0, p=norm)
      ntotal= np.empty(nx, dtype=np.int)
      for ii in range(nx):
        ntotal[ii] = len(kdtree2.query_ball_point(samplex[ii,d//2:], r=distances[ii, -1], p=norm)) - 1
      mix=-digamma(k) + digamma(nx) 
      dcmi-=p*mix
      dcmi2+=-np.sum(digamma(ntotal)) 
  return dcmi+dcmi2/n+digamma(n)

def Gao_Mixed_MI(samples,k):
  n,d=samples.shape
  xy_tree=cKDTree(samples)
  x_tree=cKDTree(samples[:,:d//2])
  y_tree=cKDTree(samples[:,d//2:])
  num=np.empty(n,dtype=np.int)
  numx=np.empty(n,dtype=np.int)
  numy=np.empty(n,dtype=np.int)
  for ii in range(n):
    dx,ix=x_tree.query(x_tree.data[ii],k=k+1)
    dy,iy=y_tree.query(y_tree.data[ii],k=k+1)
    l=max(dx[-1],dy[-1])
    if l==0.0:
      num[ii] = len(xy_tree.query_ball_point(xy_tree.data[ii], r=0.0))
    else:
      num[ii]=k
    numx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=l))
    numy[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=l)) 
  return np.mean(digamma(num))+np.log(n)-np.mean(np.log(numx)+np.log(numy))

def discrete_continuous_MI_estimate(xy,k=3):
  x,y=xy
  n=x.shape[0]
  assert n==y.shape[0]
  kdtx=KDTree(x)
  kdty=KDTree(y)
  mi=0.0
  for i in range(n):
    dx,ix=kdtx.query(x[i],k=k+1)
    dy,iy=kdty.query(y[i],k=k+1)
    rho=max(dx[-1],dy[-1])
    if rho==0.0:
      print('rho=0.0!',file=stderr); exit(1)
    else:
      ki=k
    nxi=len(kdtx.query_ball_point(x[i],rho))
    nyi=len(kdty.query_ball_point(y[i],rho))
    mi+=(psi(ki)-log(nxi)-log(nyi)-mi)/(1.0+i)
  return log(n)+mi

def Adaptive_MI(samples,k=2):
  norm=2
  n,d=samples.shape
  xy_tree=cKDTree(samples)
  x_tree  = cKDTree(samples[:,:d//2])
  y_tree  = cKDTree(samples[:,d//2:])
  LogVol=np.log(gamma(d/2+1)/(gamma(d/4+1)*gamma(d/4+1)))
  dist, idx = xy_tree.query(samples, k=k+1, p=2) 
  nx = np.empty(n, dtype=np.int)
  ny = np.empty(n, dtype=np.int)
  nxy = np.empty(n, dtype=np.int)
  for ii in range(n):
    matrix=np.c_[samples[idx[ii,:],:],np.ones(3)]
    para=np.einsum('ij,j->i',np.linalg.inv(matrix),-(samples[idx[ii,:],0]**2+samples[idx[ii,:],1]**2))
    radius=sqrt(para[0]**2+para[1]**2-4*para[2])/2
    nxy[ii] = len(xy_tree.query_ball_point(xy_tree.data[ii], r=radius, p=norm))
    nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=radius, p=norm))
    ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=radius, p=norm))
  return np.log(n)+LogVol-np.mean(np.log(nx)+np.log(ny)-np.log(nxy))

def Gknnimproved_MI(samples,k):
  n,d=samples.shape
  x_tree=cKDTree(samples)
  dist,idx=x_tree.query(samples,k=k+1,p=2)
  term4=0.0
  nx=np.empty(n,dtype=np.int)
  ny=np.empty(n,dtype=np.int)
  nz=np.empty(n,dtype=np.int) 
  for ii in range(n):
    samplex=samples[idx[ii,:],:]
    u, s, vh = np.linalg.svd(samplex-np.mean(samplex,axis=0), full_matrices=True)
    #s1=s/s[0]
    #print(np.multiply.reduce(s/s[0]))
    s=s/s[0]*dist[ii, -1]
    term4-=np.sum(np.log(s))
    samplex=(samples[idx[ii,:],:]-samples[ii,:])
    ellip=np.einsum('ij,jm,km->ik',np.diag(1.0/s),vh,samplex)
      #ellip=np.dot(np.diag(1.0/s),np.dot(vh,samplex.T))
    ellip=np.einsum('ji,ji->i',ellip,ellip)
      #ellip=np.diag(np.dot(ellip.T,ellip))
    nx[ii]=len(ellip[ellip<=1])-1
    if nx[ii]==0:
      term4+=0.0     
    else:
      term4+=np.log(nx[ii])  

    s1=s[:d//2]/s[0]*dist[ii, -1]
    #print(s1)
    term4+=np.sum(np.log(s1))
    samplex=(samples[:,:]-samples[ii,:])
    ellip=np.einsum('jm,km->kj',vh,samplex)
    #print(ellip)
    xxx=np.linalg.norm(ellip[:,:d//2]-ellip[ii,:d//2],axis=1)
    ny[ii]=len(xxx[xxx<s1])-1
    if ny[ii]==0:
      term4-=0.0     
    else:
      term4-=np.log(ny[ii])  
    s2=s[d//2:]/s[0]*dist[ii, -1]
    term4+=np.sum(np.log(s2))
    xxx2=np.linalg.norm(ellip[:,d//2:]-ellip[ii,d//2:],axis=1)
      #ellip=np.diag(np.dot(ellip.T,ellip))
    nz[ii]=len(xxx2[xxx2<s2])-1
    if nz[ii]==0:
      term4-=0.0     
    else:
      term4-=np.log(nz[ii])     
  return np.log(n)+2*np.log(np.pi**(d/4.0)/gamma(d/4.0+1.0))-np.log(np.pi**(d/2.0)/gamma(d/2.0+1.0))+term4/n#+pp/n#+np.mean(0.5/nx)#+np.mean(np.log(nx)-digamma(nx))





