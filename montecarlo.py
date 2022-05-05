from turtle import color
import numpy as np 
import math
import pandas as pd 
import plotly.graph_objects as go  
import plotly.express as px 
from plotly.subplots import make_subplots

x = 2
u = x/(2**(31))
val = []
ud = []
for i in range(1,20000):
    print(i)
    val.append((65539*x)%(2**(31)))
    x = (65539*x)%(2**(31))
    ud.append(x/(2**(31)))
    u = x/(2**(31))


u1 = []
u2 = []
for u in range(0,len(ud)-1):
    if (ud[u+1]<0.51) & (ud[u+1]>0.5): 
        u1.append(ud[u+1]) 
        u2.append(ud[u])
        
#question 2       
fig = go.Figure()
fig.add_scattergl(x=u1, y=u2)
fig.show()


u1 = []
u2 = []
u3 = []
for u in range(1,len(ud)-1):
    u1.append(ud[u+1]) 
    u2.append(ud[u])
    u3.append(ud[u-1])


#Nos 15 hyperplans (question 3 )
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=u1,y=u3,z=u2,mode='markers'))
fig.show()

#question 4

valnp = np.array(val)
valnpt = valnp % 2

valnpt.sum()

fig = px.histogram( x=valnpt , nbins = 2)
fig.show()



### EXO 2 

#loi Laplace

x = np.linspace(-10, 10, 1000)
val = 0.5*np.exp(-abs(x))
fig = go.Figure()
fig.add_scattergl(x=x, y=val)
fig.show()

#génération de point random => utilisation de l'algorithme inversion 

#FDF 


CDF1 = []
for i in x : 
    if i < 0 : 
        CDF1.append(0.5*np.exp(-abs(i)))
    else : 
        CDF1.append(1-0.5*np.exp(-abs(i))) 
       
       
fig = go.Figure()
fig.add_scattergl(x=x, y=CDF1)
fig.show()

#On calcule des point dans U de 0,1 

simU = np.random.uniform(0,1,10000)


#On construit nos points random avec l'inverse de la CDF aux points U 

rand = []
for i in simU : 
    if i - 0.5 > 0 : 
        rand.append(-1*math.log(1-2*abs(i-0.5)))
    else : 
        rand.append(math.log(1-2*abs(i-0.5)))

fig = px.histogram( x=rand )
fig.show()

#acceptation rejet 


simU = np.random.uniform(0,1,10000)
#loi normale 

X = np.random.randn(len(rand))

#ok 
fig = px.histogram( x=X )
fig.show()

valueacc = []
r = 0 
a = 0 
for i in range(0,len(rand)) : 
    if rand[i] > (1/(2*math.pi)**(0.5))*math.exp((-0.5)*(simU[i])**2) : 
        print('rejected')
        r += 1 
    else : 
        print('accepted')
        a +=1 
        valueacc.append(rand[i])

fig = px.histogram(x=valueacc)
fig.show()      
        

















# Box Muller 
simU = np.random.uniform(-1,1,10000)
simV = np.random.uniform(-1,1,10000)
X = []
for num in range(0,len(simU)):
    u = simU[num]**2
    v= simV[num]**2
    if u + v < 1 :
        S = u + v
        X.append(u*((-2*math.log(S))/S)**(0.5))
        X.append(v*((-2*math.log(S))/S)**(0.5))
    
fig = px.histogram( x=X )
fig.show()


#De 2 uniform => 2 loi Normale (0,1) iid (par Rayleigh)
import timeit
start = timeit.default_timer()

simU = np.random.uniform(0,1,10000)
simV = np.random.uniform(0,1,10000)

R = np.sqrt(-2*np.log(simU))
O = 2*np.pi*simV

#Rayleigh ? 
#fig = px.histogram( x=R )
#fig.show()


X = R*np.cos(O)
Y= R*np.sin(O)
#X 

fig = make_subplots(rows=1,cols=2)
fig.add_trace(go.Histogram(x=X),row=1,col=1)
#Y
fig.add_trace(go.Histogram(x=Y),row=1,col=2)
fig.show()

stop = timeit.default_timer()
print('Time: ', stop - start) 

#Iid ? => Identité ok 
depend = np.cov(X,Y)

#Méthode 2  => plus rapide informatiquement 


start = timeit.default_timer()
simU = np.random.uniform(0,1,10000)
simV = np.random.uniform(0,1,10000)

S = np.sqrt(-2*np.log(simU**2+simV**2))
O = np.random.uniform(0,2*math.pi,10000)

X= S*np.cos(O)
Y= S*np.sin(O)

#X 
fig = make_subplots(rows=1,cols=2)
fig.add_trace(go.Histogram(x=X),row=1,col=1)
#Y
fig.add_trace(go.Histogram(x=Y),row=1,col=2)
fig.show()

stop = timeit.default_timer()
print('Time: ', stop - start) 
#Iid ? => Identité ok 

depend = np.cov(X,Y)



#MCMC 

import numpy as np 
from scipy import stats
from matplotlib import pyplot as plt 

pi = stats.norm() #object 

x = pi.rvs(size=100)

plt.hist(x) # on va tenter de l'estimer par metropolis 
plt.show()
#fonciton 

#on applique un log car on ne calcul jamais des quantités de vraisemblance directement => on applique toujours le log (evite les problemes numériques)

sigma = 0.3
def metropolis(x):
    y = x+ sigma + np.random.randn()
    u = np.random.rand() #on simule sur une uniform 
    if np.log(u) < pi.logpdf(y) - pi.logpdf(x) : 
        return y 
    else : 
        return x
#on fait un chaine 
N = 1000
chain = np.empty(N)
chain[0] = 10.
for n in range(1,N) : 
    chain[n] = metropolis(chain[n-1])
    
plt.plot(chain[:1000])
plt.show()

#on regarde le pas de temps ou la serie devient stationnaire et on peut calculer la moyenne après cette valeur (100 a peu pres ici)
#On s'interresse à la loi donc histogram => on retrouve bien un normale (0,1)

plt.hist(chain[150:],50)
plt.show()

#on change sigma, on le baisse 
sigma =0.001 #taille des pas (plus petit) => le burning est plus long 

def metropolis(x):
    y = x+ sigma + np.random.randn()
    u = np.random.rand() #on simule sur une uniform 
    if np.log(u) < pi.logpdf(y) - pi.logpdf(x) : 
        return y 
    else : 
        return x


N = 10_000
chain = np.empty(N)
chain[0] = 10.
for n in range(1,N) : 
    chain[n] = metropolis(chain[n-1])

    
plt.plot(chain[:1000])
plt.show()

#marche pas bizarre 

#on peut faire des ACF (serie tempo) pour voir si la series généré est bien définit (indépendance)

#choix de sigma => voir slide



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





