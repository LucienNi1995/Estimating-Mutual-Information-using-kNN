# Keith Briggs & Haoran Ni 2018-06-27
# python3 Data_Generator_03.py
import numpy as np

class Data_Generator:
  ' Generate a sample from any distribution '
  def __init__(s,distribution,**kwargs):
    s.distribution=distribution
    s.kwargs=kwargs
  def sample(s,size):
    return s.distribution(size=size,**s.kwargs)

# sum of unif and stdnorm...
def sum_of_unif_stdnorm(size,alpha):
  sample=np.random.random(size)
  return np.c_[sample,np.dot(np.vstack((sample,np.random.random(size),np.random.standard_normal(size))).T,np.concatenate(([1],alpha), axis=0))]

# pareto 2d sample generator
def rpareto_inv(size,theta,a):
  return theta/np.random.random(size)**(1.0/a)

def rpareto_cond_inv(x2,theta,a):
  u=np.random.random(len(x2))
  return theta[0]+theta[0]/theta[1]*x2*(1.0/(u**(1.0/(a+1.0)))-1.0)

def pareto_2d_sample(size,theta,a):
  x2=rpareto_inv(size,theta[1],a)
  x1=rpareto_cond_inv(x2,theta,a)
  return np.vstack((x1,x2)).T

def logistic_2d_sample(size,theta,mu,a):
  u=np.random.random(size)**(-1.0/a)
  x1=mu[0]-np.log(u-1.0)*theta[0]
  x2=mu[1]-np.log(np.random.random(size)**(-0.5/a)*u-u)*theta[1]
  return np.vstack((x1,x2)).T
#def logistic_2d_sample(size,theta,mu,a):
 # x2=np.random.logistic(loc=theta[1], scale=mu[1], size=size)
  #randcdf=np.random.random(size)
  #for i in range(size):
   # while (randcdf[i]==0.0): randcdf[i]=np.random.random(1)
  #x1=theta[0]-np.log((1.0/randcdf**0.5-1.0)*(1+np.exp(-(x2-mu[1])/theta[1])))*mu[0]
  #return np.vstack((x1,x2)).T

def burr_2d_sample(size,d,c,a):
  u=np.random.random(size)**(-1.0/a)
  x1=((u-1.0)/d[0])**(1.0/c[0])
  x2=((np.random.random(size)**(-1.0/a)*u-u)/d[1])**(1.0/c[1])
  return np.vstack((x1,x2)).T

# written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
  '''generate random variables of multivariate t distribution
  Parameters
  ----------
  m : array_like
        mean of random variable, length determines dimension of random variable
  S : array_like
      square array of covariance  matrix
  df : int or float
      degrees of freedom
  n : int
      number of observations, return random array will be (n, len(m))
  Returns
  -------
  rvs : ndarray, (n, len(m))
      each row is an independent draw of a multivariate t distributed
      random variable
  '''
  m = np.asarray(m)
  d = len(m)
  if df == np.inf:
    x = 1.
  else:
    x = np.random.chisquare(df, n)/df
  z = np.random.multivariate_normal(np.zeros(d),S,(n,))
  return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

def mapping_vec(z):
  return np.c_[z.real,z.imag]

def mapping_mat(A):
  return np.c_[np.r_[A.real,A.imag],np.r_[-A.imag,A.real]]

def remapping(z):
  return z[:,:len(z[0])//2]+z[:,len(z[0])//2:]*1j

def complex_gaussian(size,mean,H,Q):
  H_hat=mapping_mat(H)
  mean_hat=np.r_[mean.real,mean.imag]
  cov=0.5*mapping_mat(Q)
  X=np.random.multivariate_normal(size=size,mean=mean_hat,cov=cov)
  I_r=np.eye(len(H),dtype='float')
  cov_n=0.5*mapping_mat(I_r)
  mean_n=np.zeros(len(H_hat),dtype='float')
  n=np.random.multivariate_normal(size=size,mean=mean_n,cov=cov_n)
  Y=np.empty(X.shape)
  for ii in range(size):
    Y[ii,:]=np.einsum('ij,j->i',H_hat,X[ii,:])+n[ii,:]
  return np.c_[X,Y]

def Gibbs_sampling(samplesize,d,num_ini):
  #initialize samples matrix
  samples=np.empty((samplesize,d),dtype='float')
  #initialize the first sample
  burn_in=np.ones(d,dtype='float') 
  #burn_in
  for ii in range(num_ini):
    burn_in2=con_pdf(burn_in)
  #sampling
  samples[0,:]=burn_in2
  for jj in range(samplesize):
    samples[jj+1,:]=samples[jj,:]
    samples[jj+1,:]=con_pdf(samples[jj,:])
  return samples


if __name__=='__main__': # self-test
  import numpy as np
  # uniform...
  dg=Data_Generator(np.random.random)
  samples=dg.sample(size=10)
  # print(samples)
  # normal...
  dg=Data_Generator(np.random.normal,loc=0.0,scale=1.0)
  samples=dg.sample(size=10000)
  # print(np.var(samples))
  # print(samples)
  # multivariate_normal...
  mean = np.array([1.0,-2.0,1.0,-2.0])
  cov  = np.array([[10.0,1.0e-1,1.0e-1,1.0e-1],[1.0e-1,5.0,1.0e-1,1.0e-1],[1.0e-1,1.0e-1,5.0,1.0e-1],[1.0e-1,1.0e-1,1.0e-1,5.0]])
  dg=Data_Generator(np.random.multivariate_normal,mean=mean,cov=cov)
  samples=dg.sample(size=10000)
  # print(np.cov(samples[:,0:2].T,samples[:,2:4].T))
  # print(cov)
  # print(samples)
  # sum of uniform...
  dg=Data_Generator(sum_of_unif_stdnorm,alpha=[0.5,0])
  samples=dg.sample(size=10)
  # print(samples)
  # sum of unif and std norm...
  dg=Data_Generator(sum_of_unif_stdnorm,alpha=[0,1.0])
  samples=dg.sample(size=10)
  # print(samples)
  theta=np.array([5.0,2.0])
  a=3.0
  dg=Data_Generator(pareto_2d_sample,theta=theta,a=a)
  samples=dg.sample(size=10)
  #print(samples)

