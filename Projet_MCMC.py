#test MCMC sans biais 

#Val initiales : Jeu de donnée et Z_0 

import numpy as np
from scipy import stats
from scipy.stats import invgamma
from scipy.special import gamma
import plotly as plt 
import plotly.graph_objects as go


###Probability density function of inverse gamma distribution###
class invgamma(stats.rv_continuous):
    def _pdf(self, x,alpha,beta):
        px = (beta**alpha)/gamma(alpha)*x**(-alpha-1)*np.exp(-beta/x)
        return px

###Sampling from inverse gamma function###
invgamma = invgamma(name="invgamma", a=0.0)    
sample_from_invgamma = invgamma.rvs(size=1, alpha = 1, beta = 2)


Z = np.array([0.395, 0.375, 0.355, 0.334, 0.313, 0.313, 0.291, 0.269, 0.247, 0.247, 0.224, 0.224, 0.224, 0.224, 0.224, 0.200, 0.175, 0.148])

#Gibbs algo 

def Gibbs(N,Z):
    #initiation
    V = 0.00434
    the = np.empty((N+1,len(Z)))
    A = np.empty((N+1,1))
    mu = np.empty((N+1,1))
    the[0] = Z.mean()
    A[0] =  invgamma.rvs(size=1, alpha = 7.5, beta = 2 + np.var(the[0])/2)
    mu[0] = np.random.normal(loc = the[0].mean(),scale = A[0]/len(Z))
    for k in range (0,N,1):
        the[k+1] = np.random.normal(loc = ((mu[k]*V+Z*A[k])/(V+A[k])),scale = (A[k]*V)/(A[k]+V))
        A[k+1] =  invgamma.rvs(size=1, alpha = 7.5, beta = 2 + np.var(the[k])/2)
        mu[k+1] = np.random.normal(loc = the[k].mean(),scale = A[k]/len(Z))
    return(the)

    
sample = Gibbs(500,Z)

df = pd.DataFrame(sample)

fig = go.Figure()  
for i in range(0,N,1):
    fig.add_scattergl(x=list(range(1,len(Z)+1)), y=sample[i])
fig.add_scattergl(x=list(range(1,len(Z)+1)), y=Z,line={'color': 'black'})
fig.show()