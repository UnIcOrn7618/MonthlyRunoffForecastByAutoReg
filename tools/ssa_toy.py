#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcdwt(), 'tools'))
	print(os.getcdwt())
except:
	pass

#%%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=14
plt.rcParams['image.cmap']='plasma'
plt.rcParams['axes.linewidth']=2

from cycler import cycler
cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle']=cycler(color=cols)

def plot_2d(m,title=''):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
# Generate a toy series
# f=0.001*(t-100)^2+2*sin(2*pi*t/p1)+0.75sin(2*pi*t/p2)+Rand{-1,1}    
N=200
t=np.arange(0,N)
trend=0.001*(t-100)**2
p1,p2=20,30
periodic1=2*np.sin(2*pi*t/p1)
periodic2=0.75*np.sin(2*pi*t/p2)

np.random.seed(123)
noise=2*(np.random.rand(N)-0.5)
F = trend+periodic1+periodic2+noise

plt.plot(t,F,lw=2.5)
plt.plot(t,trend,alpha=0.75)
plt.plot(t,periodic1,alpha=0.75)
plt.plot(t,periodic2,alpha=0.75)
plt.plot(t,noise,alpha=0.75)
plt.legend(['Toy Series ($F$)','Trend','Periodic #1','Periodic #2','Noise'])
plt.xlabel('$t$')
plt.ylabel('$F(t)$')
plt.title('The Toy Time Series and its Components')
plt.show()

#%%

L=70 # The window length.
K=N-L+1 # The number of columns in the trajectory matrix
X=np.column_stack([F[i:i+L] for i in range(0,K)])

ax = plt.matshow(X)
plt.xlabel('$L$-Lagged Vectors')
plt.ylabel('$K$-Lagged Vectors')
plt.colorbar(ax.colorbar,fraction=0.025)
ax.colorbar.set_label('$F(t)$')
plt.title('The Trajectory Matrix for the Toy Time Series')
plt.show()

#%%
d = np.linalg.matrix_rank(X) # The intrinsic dimensionality of the trajectory space.
U, Sigma, V = np.linalg.svd(X)
V=V.T
X_elem = np.array([Sigma[i]*np.outer(U[:,i],V[:,i]) for i in range(0,d)])
if not np.allclose(X,X_elem.sum(axis=0),atol=1e-10):
    print("WARNING: The sum of X's elementary matrices is not equal to X!")

n=min(12,d) # In case d is less than 12 for the toy series. Say if we were to exclude the noise component
for i in range(n):
    plt.subplot(4,4,i+1)
    title='$\mathbf{X}_{'+str(i)+'}$'
    plot_2d(X_elem[i],title)
plt.tight_layout()
plt.show()

#%%
# plot the relative contributions and the cumulative contributions
sigma_sumsq = (Sigma**2).sum()
fig,ax = plt.subplots(1,2,figsize=(14,5))
ax[0].plot(Sigma**2/sigma_sumsq*100,lw=2.5)
ax[0].set_xlim(0,11)
ax[0].set_title("Relative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
ax[0].set_xlabel('$i$')
ax[0].set_ylabel('Contribution (%)')
ax[1].plot((Sigma**2).cumsum()/sigma_sumsq*100,lw=2.5)
ax[1].set_xlim(0,11)
ax[1].set_title("Cumulative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
ax[1].set_xlabel('$i$')
ax[1].set_ylabel('Contribution (%)')


#%%
def Hankelise(X):
    """
    Hankelises the matrix X, returning H(X).
    """
    L, K = X.shape
    transpose = False
    if L > K:
        # The Hankelisation below only works for matrices where L < K.
        # To Hankelise a L > K matrix, first swap L and K and tranpose X.
        # Set flag for HX to be transposed before returning. 
        X = X.T
        L, K = K, L
        transpose = True

    HX = np.zeros((L,K))
    
    # I know this isn't very efficient...
    for m in range(L):
        for n in range(K):
            s = m+n
            if 0 <= s <= L-1:
                for l in range(0,s+1):
                    HX[m,n] += 1/(s+1)*X[l, s-l]    
            elif L <= s <= K-1:
                for l in range(0,L-1):
                    HX[m,n] += 1/(L-1)*X[l, s-l]
            elif K <= s <= K+L-2:
                for l in range(s-K+1,L):
                    HX[m,n] += 1/(K+L-s-1)*X[l, s-l]
    if transpose:
        return HX.T
    else:
        return HX

n = min(d, 12)
for j in range(0,n):
    plt.subplot(4,4,j+1)
    title = r"$\tilde{\mathbf{X}}_{" + str(j) + "}$"
    plot_2d(Hankelise(X_elem[j]), title)
plt.tight_layout() 

#%%
def X_to_TS(X_i):
    """Averages the anti-diagonals of the given elementary matrix,
     X_i, and return a time series"""
    #  Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])


n=min(12,d)
fig=plt.subplot()
color_cycle = cycler(color=plt.get_cmap('tab20').colors)
fig.axes.set_prop_cycle(color_cycle)

for i in range(n):
    F_i = X_to_TS(X_elem[i])
    fig.axes.plot(t,F_i,lw=2)

fig.axes.plot(t,F,alpha=1,lw=1)
fig.set_xlabel('$t$')
fig.set_ylabel(r"$\tilde{F}_i(t)$")
legend=[r"$\tilde{F}_{%s}$" %i for i in range(n)] + ["$F$"]
fig.set_title("The first 12 components of the Toy Time Series")
fig.legend(legend,loc=(1.05,0.1))

#%%
# Assemble the grouped components of the time series
F_trend = X_to_TS(X_elem[[0,1,6]].sum(axis=0))
F_periodic1 = X_to_TS(X_elem[[2,3]].sum(axis=0))
F_periodic2 = X_to_TS(X_elem[[4,5]].sum(axis=0))
F_noise = X_to_TS(X_elem[7:].sum(axis=0))

# Plot the toy series and its separated components on a single plot
plt.plot(t,F,lw=1)
plt.plot(t,F_trend)
plt.plot(t,F_periodic1)
plt.plot(t,F_periodic2)
plt.plot(t,F_noise,alpha=0.5)
plt.xlabel('$t$')
plt.ylabel(r'$\tilde{F}^{(j)}$')
groups = ['trend','periodic 1','periodic 2','noise']
legend = ["$F$"]+[r"$\tilde{F}^{(\mathrm{%s})}$"% group for group in groups]
plt.legend(legend)
plt.title("Grouped Time Series Components")
# A list of tuples
components = [
    ("Trend",trend,F_trend),
    ("Periodic 1",periodic1,F_periodic1),
    ("Periodic 2",periodic2,F_periodic2),
    ("Noise",noise,F_noise),
]
# plot the separated components and original components together
fig = plt.figure()
n = 1
for name,orig_comp,ssa_comp in components:
    ax = fig.add_subplot(2,2,n)
    ax.plot(t,orig_comp,linestyle='--',lw=2.5,alpha=0.7)
    ax.plot(t,ssa_comp)
    ax.set_title(name,fontsize=16)
    ax.set_xticks([])
    n+=1
fig.tight_layout()

#%%
# Get the weights w first, as they'll be reused a lot.
# Note: list(np.arange(L)+1) return the sequence 1 to L (first line in definition of W),
# [L]*(K-L-1) repeats L K-L-1 times (second line in w difinition)
# list(np.arange(L)+1)[::-1] reverses the first list (equivalent to the third line)
# add all the lists together and we have our array of weights.
w = np.array(list(np.arange(L)+1) +[L]*(K-L-1) + list(np.arange(L)+1)[::-1])

# Get all the copmponents of the toy series, store them as columns in F_elem array
F_elem = np.array([X_to_TS(X_elem[i]) for i in range(d)])

# Calculate the individual weighted norms, ||F_i||_w, first, then take inverse
# square-root  so we don't have to later.
F_wnorms = np.array([w.dot(F_elem[i]**2) for i in range(d)])
F_wnorms = F_wnorms**-0.5

# calculate the w-corr matrix, The diagonal elements are equal to 1,
# so we can start with an identity matrix 
# and isterate over all pairs of i's and j's (i!=j), nothing that Wij=Wji
Wcorr = np.identity(d)
for i in range(d):
    for j in range(i+1,d):
        Wcorr[i,j] = abs(w.dot(F_elem[i]*F_elem[j])*F_wnorms[i]*F_wnorms[j])
        Wcorr[j,i] = Wcorr[i,j]

# plot the w-correlation matrix
ax = plt.imshow(Wcorr)
plt.xlabel(r"$\tilde{F}_i$")
plt.ylabel(r"$\tilde{F}_j$")
plt.colorbar(ax.colorbar,fraction=0.045)
ax.colorbar.set_label("$W_{ij}$")
plt.clim(0,1)
plt.title("The W-Correlation Matrix for the Toy Time Series")


#%%
ax = plt.imshow(Wcorr)
plt.xlabel(r"$\tilde{F}_i$")
plt.ylabel(r"$\tilde{F}_j$")
plt.colorbar(ax.colorbar,fraction=0.045)
ax.colorbar.set_label("$W_{ij}$")
plt.xlim(-0.5,6.5)
plt.ylim(6.5,-0.5)
plt.clim(0,1)
plt.title("The W-Correlation Matrix for Components 0-6")

#%%
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
parent_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
grandpa_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
data_path = parent_path + '\\data\\'

import sys
sys.path.append(grandpa_path+'/tools/')
from ssa import SSA
F_ssa_L2 = SSA(F,2)
F_ssa_L2.components_to_df().plot()
F_ssa_L2.orig_TS.plot(alpha=0.4)
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.title(r"$L=2$ for the Toy Time Series")

#%%
F_ssa_L5 = SSA(F,5)
F_ssa_L5.components_to_df().plot()
F_ssa_L5.orig_TS.plot(alpha=0.4)
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.title(r"$L=5$ for the Toy Time Series")

#%%
F_ssa_L20 = SSA(F,20)
F_ssa_L20.plot_wcorr()
plt.title("W-Correlation for Toy Time Series, $L=20$")

#%%
F_ssa_L20.reconstruct(0).plot()
F_ssa_L20.reconstruct([1,2,3]).plot()
F_ssa_L20.reconstruct(slice(4,20)).plot()
F_ssa_L20.reconstruct(3).plot()
plt.xlabel("$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.title("Component Groupings for Toy Time Series, $L=20$")
plt.legend([
    r"$\tilde{F}_0$",
    r"$\tilde{F}_1+\tilde{F}_2+\tilde{F}_3$",
    r"$\tilde{F}_4+\ldots+\tilde{F}_{19}$",
    r"$\tilde{F}_3$",
])

#%%
F_ssa_L40 = SSA(F,40)
F_ssa_L40.plot_wcorr()
plt.title("W-Correlation for Toy Time Series, $L=40$")

#%%
F_ssa_L40.reconstruct(0).plot()
F_ssa_L40.reconstruct([1,2,3]).plot()
F_ssa_L40.reconstruct([4,5]).plot()
F_ssa_L40.reconstruct(slice(6,40)).plot(alpha=0.7)
plt.title("Component Groupings for Toy Time Series, $L=40$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\tilde{F}_i(t)$")
plt.legend([r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(4)])

#%%
F_ssa_L60=SSA(F,60)
F_ssa_L60.plot_wcorr()
plt.title("W-Correlation for Toy Time Series, $L=60$")

#%%
F_ssa_L60.reconstruct(slice(0,7)).plot()
F_ssa_L60.reconstruct(slice(7,60)).plot()
plt.legend([
    r"$\tilde{F}^{\mathrm{(signal)}}$",
    r"$\tilde{F}^{\mathrm{(noise)}}$",
])
plt.title("Signal and Noise Components of Toy Time Series, $L=60$")
plt.xlabel(r"$t$")

#%%
F_ssa_L60.plot_wcorr(max=6)
plt.title("W-Correlation for Toy Time Series, $L=60$ (Zoomed)")

#%%
F_ssa_L60.components_to_df(n=7).plot()
plt.title(r"The First 7 Components of the Toy Time Series, $L=60$")
plt.xlabel(r"$t$")

#%%
