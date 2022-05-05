#test MCMC sans biais 


#question 1 


#packages

import numpy as np
from scipy import stats
from scipy.stats import invgamma
from scipy.special import gamma
from scipy.stats import gamma
import plotly as plt 
import plotly.graph_objects as go
import pandas as pd 
import random 
import plotly.express as px 
from scipy.stats import norm
import numpy as np
import scipy.stats as st


###Probability density function of inverse gamma distribution###
class invgamma(stats.rv_continuous):
    def _pdf(self, x,alpha,beta):
        px = (beta**alpha)/gamma(alpha)*x**(-alpha-1)*np.exp(-beta/x)
        return px

###Sampling from inverse gamma function###
invgamma = invgamma(name="invgamma", a=0.0)    

Z = np.array([0.395, 0.375, 0.355, 0.334, 0.313, 0.313, 0.291, 0.269, 0.247, 0.247, 0.224, 0.224, 0.224, 0.224, 0.224, 0.200, 0.175, 0.148])

test = [0.395, 0.375, 0.355, 0.334, 0.313, 0.313, 0.291, 0.269, 0.247, 0.247, 0.224, 0.224, 0.224, 0.224, 0.224, 0.200, 0.175, 0.148]

#Gibbs algo 


V = 0.00434  #article value V = 0.00434 

def Gibbs(N,Z,V):
    #initiation
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
    return([the,A,mu])

N = 1000 #nombre de sample à produire
sample = Gibbs(N,Z,V)

df1 = pd.DataFrame(sample[1])
df2 = pd.DataFrame(sample[2])
df = pd.DataFrame(sample[0]) 


def graph():
    fig = go.Figure()  
    t = []
    for col in df.columns : 
        t.append(df[col].mean())
    for i in range(0,N,1):
        fig.add_scattergl(x=list(range(1,len(Z)+1)), y=sample[i])
    fig.add_scattergl(x=list(range(1,len(Z)+1)), y=t,line={'color': 'black'})
    fig.add_scattergl(x=list(range(1,len(Z)+1)), y=Z,line={'color': 'red'})
    fig.show()

graph()


fig = go.Figure()
fig.add_scattergl(x=list(range(1,len(Z)+1)), y=t,line={'color': 'black'})
fig.add_scattergl(x=list(range(1,len(Z)+1)), y=Z,line={'color': 'red'})
fig.show()


fig = px.histogram(df1, x=0) 
fig.show()

fig = px.histogram(df2, x=0) 
fig.show()


#question 2 


#avec 1 gamma(alpha = 1) et une normale(1,1)

def maximal_coupling():
    X = gamma.rvs(1)
    W = norm.rvs(loc=0,scale=1) * gamma.pdf(X,a=1)
    if W < norm.pdf(X,loc=1,scale=1):
        return X, X
    else:
        while True:
            Y = norm.rvs(loc=1,scale=1)
            W = norm.rvs(loc=0,scale=1) * norm.pdf(Y,loc=1,scale=1)
            if W > gamma.pdf(Y,a=1):
                return [X,Y]


listXY = pd.DataFrame([maximal_coupling() for _ in range (10000)])


plot = px.Figure(data=[px.Scatter( x = listXY[0], y = listXY[1], mode = 'markers',) ]) 
plot.show()


#test avec 2 normales


def maximal_coupling2():
    X = norm.rvs(loc=0,scale=1)
    W = norm.rvs(loc=0,scale=1) * norm.pdf(X,loc=0,scale=1)
    if W < norm.pdf(X,loc=0,scale=1):
        return X, X
    else:
        while True:
            Y = norm.rvs(loc=0,scale=1)
            W = norm.rvs(loc=0,scale=1) * norm.pdf(Y,loc=0,scale=1)
            if W > norm.pdf(Y,loc=0,scale=1):
                return [X,Y]


listXY = pd.DataFrame([maximal_coupling2() for _ in range (10000)])


plot = px.Figure(data=[px.Scatter( x = listXY[0], y = listXY[1], mode = 'markers',) ]) 
plot.show()




        

#question 3 ? 




def metropolis_hastings_vec(log_prob, proposal_cov, iters, chains, init):
    """Vectorized Metropolis-Hastings.
    Allows pretty ridiculous scaling across chains:
    Runs 1,000 chains of 1,000 iterations each on a
    correlated 100D normal in ~5 seconds.
    """
    proposal_cov = np.atleast_2d(proposal_cov)
    dim = proposal_cov.shape[0]
    # Initialize with a single point, or an array of shape (chains, dim)
    if init.shape == (dim,):
        init = np.tile(init, (chains, 1))
    samples = np.empty((iters, chains, dim))
    samples[0] = init
    current_log_prob = log_prob(init)
    proposals = np.random.multivariate_normal(np.zeros(dim), proposal_cov,
                                              size=(iters - 1, chains))
    log_unifs = np.log(np.random.rand(iters - 1, chains))
    for idx, (sample, log_unif) in enumerate(zip(proposals, log_unifs), 1):
        proposal = sample + samples[idx - 1]
        proposal_log_prob = log_prob(proposal)
        accept = (log_unif < proposal_log_prob - current_log_prob)
        # copy previous row, update accepted indexes
        samples[idx] = samples[idx - 1]
        samples[idx][accept] = proposal[accept]
        # update log probability
        current_log_prob[accept] = proposal_log_prob[accept]
    return samples


dim = 10
Σ = 0.1 * np.eye(dim) + 0.9 * np.ones((dim, dim))

# Correlated Gaussian
log_prob = st.multivariate_normal(np.zeros(dim),Σ).logpdf

proposal_cov = np.eye(dim)
iters = 2_000
chains= 1_024
init = np.zeros(dim)

samples = metropolis_hastings_vec(log_prob, proposal_cov, iters, chains, init)


