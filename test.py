#%% imports
import numpy as np
from matplotlib import pyplot as plt
import copy
import pdb
#%% 1-1,1-2
N=128
a=1
b=5
U = np.zeros(N)
U[0] = 0
for i in range(len(U)):
    if 1 == 0:
        pass
    else:
        U[i] = np.random.normal(U[i-1],a)

V = np.zeros(N)
for i in range(len(V)):
    V[i] = np.random.normal(U[i],b)

#%%
plt.plot(U)
plt.plot(V)
plt.show()
# %% 1-3 SDG
def gradE(U,V,i,lam):
    if i==0:
        return U[i]-V[i]+lam*(U[i]-U[i+1])
    elif i==len(U)-1:
        return U[i]-V[i]+lam*(U[i]-U[i-1])
    else:
        return U[i]-V[i]+lam*(2*U[i]-(U[i-1]+lam*U[i+1]))

def threshold(A_1,A_2):
    #return np.abs((A_1 - A_2)).mean()
    return np.abs((A_1 - A_2)).max()

def SGD(V,lam,k):
    U_old = np.random.rand(N)
    U_new = np.random.rand(N) 
    t=0
    while(threshold(U_new,U_old) > 0.00001):
        t=t+1
        U_old = copy.copy(U_new)
        for i in range(len(U)):
            U_new[i] = U_new[i] - k*gradE(U_new,V,i,lam)
    print("done with {}times iteration".format(t))
    return U_new

#%%
test_U_new = SGD(V,lam=1.3,k=0.05)
plt.plot(U,label="True image",c='r')
plt.plot(test_U_new,label="Estimated image",c='g')
plt.plot(V,label="Obsereved image",c='b')
plt.legend(loc='upper left')
plt.show()
print("min: {}".format(test_U_new.min()))
print("max: {}".format(test_U_new.max()))
print("mean:{}".format(test_U_new.mean()))

# %%
sdf