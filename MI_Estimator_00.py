# Keith Briggs & Haoran Ni 2018-06-27
# python3 MI_Estimator_00.py

import scipy.integrate as integrate
import scipy.special as special
import numpy as np
import Methods_Collection_00 as mtdc
from scipy.special import digamma
from numpy import ndarray
from scipy.stats import multivariate_normal


class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T

class MI_Estimator:
  ' Apply MI_Estimator for samples.'
  def __init__(s, method,**params):
    s.methods=method
    s.params=params
  def estimate_MI(s,samples):
    return s.methods(samples=samples,**s.params)

class Exact:
  'Compute the exact solutions from known forms'
  #three families
  def __init__(s, value=None):
    s.value=value
  def model1(s,alpha):
    ' Compute the MI of Model1 '
    return -np.log(alpha)
  def model2(s,alpha):
    ' Compute the MI of Model2 '
    yxx= lambda y:special.ndtr(1-y/alpha)-special.ndtr(-y/alpha)
    fxx=integrate.quad(lambda y:-yxx(y)*np.log(yxx(y)),-alpha,alpha)
    return -np.log(alpha)+fxx[0]-0.5*np.log(2.0*np.pi*np.e)    
  def model3(s,cov):
    ' Compute the MI of Model3 '
    n=cov.shape[0]
    return 0.5*np.log(np.linalg.det(cov[0:n//2,0:n//2])*np.linalg.det(cov[n//2:n,n//2:n])/np.linalg.det(cov))

  # special distributuion
  def pareto_2d(s,a):
    ' Compute the MI of the 2d pareto distribution'
    return np.log(1.0+1.0/a)-1.0/(a+1.0)
  def pareto_nd(s,n,a):
    ' Compute the MI of the nd pareto distribution'
    sum1=0.0
    sum2=0.0
    for ii in range(n//2):
      sum1+=1.0/(a+ii)
      sum2+=1.0/(a+ii+n/2.0)
    return np.log(n-2.0+4.0*a)-np.log(3.0*n-2.0+4.0*a)+a*sum1-(a+n)*sum2
  def ordered_Weinmanexp_2d(s,theta):
    ' Compute the MI of the 2d ordered Weinman exponential distribution'
    t=theta[0]-2*theta[1]
    if t>0.0:
      return np.log(0.5*t/theta[1])+digamma(theta[0]/t)+digamma(1.0)
    elif t<0.0:
      return np.log(-0.5*t/theta[1])+digamma(-2.0*theta[1]/t)+digamma(1.0)
    else:
      return digamma(1.0)
  def Gamma_exp(s,theta2):
    ' Compute the MI of the Gamma exponential distribution'
    return digamma(theta2)-np.log(theta2)+1.0/theta2


  #exact solution of 4QAM with Gaussian
  def dc_4QAM(s,mu,sigma,pb):
    n=sigma.shape[0]
    d=sigma.shape[2]
    dets=np.array([np.linalg.det(sigma[i,:,:]) for i in range(n)])
    entropy1=np.sum(pb*0.5*np.log((2*np.pi*np.e)**d*dets))
    func=lambda y,x: -integrand(x,y,mu,sigma,pb)*np.log(integrand(x,y,mu,sigma,pb))
    evals=np.vstack([np.linalg.eigvals(sigma[0]),np.linalg.eigvals(sigma[1]),np.linalg.eigvals(sigma[2]),np.linalg.eigvals(sigma[3])])
    a,b=-5*(np.max(np.abs(evals))+np.mean(mu[:,0])),5*(np.max(np.abs(evals))+np.mean(mu[:,1]))
    entropy2=integrate.dblquad(func,a,b,lambda x: a, lambda x: b)
    return entropy2[0]-entropy1,entropy2[1]




  #complex gaussian
  def cpx_gaussian(s,H,Q):
    I_t=np.zeros(Q.shape)
    for ii in range(len(H[0])):
      I_t[ii,ii]=1.0
    return np.log(np.linalg.det(I_t+np.einsum('ij,kj,km->im',Q,np.conj(H),H))).real

def generate_Q(d):
  A=(np.random.random((d,d))+np.random.random((d,d))*1j).view(myarray)
  return np.einsum('ij,jk->ik',A.H,A)

def integrand(x,y,mu,sigma,pb):
  return pb[0]*multivariate_normal.pdf([x,y],mean=mu[0,:], cov=sigma[0,:,:])+ pb[1]*multivariate_normal.pdf([x,y],mean=mu[1,:], cov=sigma[1,:,:])+pb[2]*multivariate_normal.pdf([x,y],mean=mu[2,:], cov=sigma[2,:,:])+pb[3]*multivariate_normal.pdf([x,y],mean=mu[3,:], cov=sigma[3,:,:])

def repeat2(ii):
  samplesize=10000
  error=0.1
  k=ii+1
  #kgnn=20
  mean = np.empty(5)
  svar =np.empty(5)
  mean[0],svar[0]=mean_std('threeKL_MI',samplesize,k,error,dg,x)
  mean[1],svar[1]=mean_std('KSG1_MI',samplesize,k,error,dg,x)
  mean[2],svar[2]=mean_std('KSG2_MI',samplesize,k,error,dg,x)
  mean[3],svar[3]=mean_std('BIKSG_MI',samplesize,k,error,dg,x)
  #mean[4],svar[4]=mean_std('Gknn_MI',samplesize,kgnn,error,dg,x)
  f = open("N_error_std1k","a")
  f.write('%g\t'%k)
  f.write('%g\t%g\t%g\t%g\t%g\t'%tuple(mean))
  f.write('%g\t%g\t%g\t%g\t%g\t'%tuple(np.sqrt(svar)))
  f.write("\n")
  f.close() 

def repeat(ii):
  samplesize=np.int(10**((ii+4)/3))
  error=0.1
  k=3
  kgnn=20
  mean = np.empty(4)
  svar =np.empty(4)
  mean[0],svar[0]=mean_std('threeKL_MI',samplesize,k,error,dg,x)
  mean[1],svar[1]=mean_std('KSG1_MI',samplesize,k,error,dg,x)
  mean[2],svar[2]=mean_std('KSG2_MI',samplesize,k,error,dg,x)
  mean[3],svar[3]=mean_std('BIKSG_MI',samplesize,k,error,dg,x)
  #mean[4],svar[4]=mean_std('Gknn_MI',samplesize,kgnn,error,dg,x)
  f = open("N_error_std1","a")
  f.write('%g\t'%samplesize)
  f.write('%g\t%g\t%g\t%g\t'%tuple(mean))
  f.write('%g\t%g\t%g\t%g\t'%tuple(np.sqrt(svar)))
  f.write("\n")
  f.close() 

def mean_std(methods,samplesize,k,error,dg,x):
  mean=0.0
  svar=0.0
  for N in range(1000-samplesize//100+1):
    #np.random.seed(seed=N)
    samples=dg.sample(size=samplesize) 
    if methods=='BIKSG_MI':
      miesti=MI_Estimator(mtdc.BIKSG_MI,k=k,norm=2)
    elif methods=='KSG1_MI':
      miesti=MI_Estimator(mtdc.KSG1_MI,k=k,norm=np.inf)
    elif methods=='KSG2_MI':
      miesti=MI_Estimator(mtdc.KSG2_MI,k=k,norm=np.inf)
    elif methods=='threeKL_MI':
      miesti=MI_Estimator(mtdc.threeKL_MI,k=k,norm=np.inf)
    elif methods=='Gknnimproved_MI':
      miesti=MI_Estimator(mtdc.Gknnimproved_MI,k=k)
    else:
      miesti=MI_Estimator(mtdc.Gknn_MI,k=k)
    MI=miesti.estimate_MI(samples)-x
    oldmean=mean
    mean+=(MI-mean)/(1.0+N)
    svar+=(MI-mean)*(MI-oldmean)
    #if N>10:
      #stder=np.sqrt(svar/(1+N))
      #if stder<error*(1.0+abs(mean)): break
        #if stder/np.sqrt(N)<error: break
 # if x==0: 
  return mean,svar
 # else:
  #  return mean/x,svar
   
def repeat3(ii):
  samplesize=np.int(10**((ii+5)/3))
  error=0.1
  k=3
  kgnn=20
  mean = np.empty(5)
  svar =np.empty(5)
  mean[0],svar[0]=mean_std('threeKL_MI',samplesize,k,error,dg,x)
  mean[1],svar[1]=mean_std('KSG1_MI',samplesize,k,error,dg,x)
  mean[2],svar[2]=mean_std('KSG2_MI',samplesize,k,error,dg,x)
  mean[3],svar[3]=mean_std('BIKSG_MI',samplesize,k,error,dg,x)
  mean[4],svar[4]=mean_std('Gknn_MI',samplesize,kgnn,error,dg,x)
  f = open("N_logstic","a")
  f.write('%g\t'%samplesize)
  f.write('%g\t%g\t%g\t%g\t%g\t'%tuple(mean))
  f.write('%g\t%g\t%g\t%g\t%g\t'%tuple(np.sqrt(svar)))
  f.write("\n")
  f.close() 

if __name__=='__main__': # self-test
 
  import Data_Generator_03 as Dtg
  import Methods_Collection_00 as mtdc
  import scipy.stats as stats
  import numpy as np
  import multiprocessing

  #multigaussian
  #np.random.seed(seed=None)
  #mean = np.array([0,0.0])
 # for rr in range (0,4):
    #cov  = np.array([[1,rr*0.3],[rr*0.3,1.0]])
    #a=Exact()
    #x=a.model3(cov)
    #dg=Dtg.Data_Generator(np.random.multivariate_normal,mean=mean,cov=cov)
    #pint=14
    #cores=multiprocessing.cpu_count()
    #pool=multiprocessing.Pool(processes=cores)
    #ii=list(range(pint))
    #pool.map(repeat,ii)
  #multigaussiank
  #np.random.seed(seed=None)
  #mean = np.array([0,0.0])
  #rr=3
  #cov  = np.array([[1,rr*0.3],[rr*0.3,1.0]])
  #a=Exact()
  #x=a.model3(cov)
  #dg=Dtg.Data_Generator(np.random.multivariate_normal,mean=mean,cov=cov)
  #pint=36
  #cores=multiprocessing.cpu_count()
  #pool=multiprocessing.Pool(processes=cores)
  #ii=list(range(pint))
  #pool.map(repeat2,ii)
 #pareto
  theta = np.array([2.0,5.0])
  mu = np.array([0,0.0])
  a=1.0
  d=Exact()
  x=d.pareto_2d(a)
  print(x)
  #dg=Dtg.Data_Generator(Dtg.pareto_2d_sample,theta=theta,a=a)
  dg=Dtg.Data_Generator(Dtg.logistic_2d_sample,theta=theta,mu=mu,a=a)
  #dg=Dtg.Data_Generator(Dtg.burr_2d_sample,d=theta,c=mu,a=a)
  pint=14
  cores=multiprocessing.cpu_count()
  pool=multiprocessing.Pool(processes=cores)
  ii=list(range(pint))
  pool.map(repeat3,ii)





