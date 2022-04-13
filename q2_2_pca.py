import numpy as np
import scipy.io
from numpy import linalg as LA
from matplotlib import pyplot as plt

mat = scipy.io.loadmat('USPS.mat')

P=[10,50,100,200]

X=mat['A']
Y=mat['L']

X_centered = X - np.mean(X , axis = 0)
     
cov= np.cov(X_centered , rowvar = False)
    
#### Performing SVD ####    
lambd,v = np.linalg.eigh(cov)

## Sorting the eigen values and vectors in descending order ###
indexes = np.argsort(lambd)[::-1]
lambd_s = lambd[indexes]
v_s = v[:,indexes]

reconstrucion_errors=[]

for p in P:
    ### Select the p components ###    
    selected_v = v_s[:,0:p]
    selected_lambds=lambd_s[0:p]

    X_pca=np.dot(X_centered,selected_v)
    diag=np.diag(selected_lambds)

    transformed_X=np.dot(X_pca,selected_v.transpose())

    recon_error=(np.linalg.norm(X_centered-transformed_X, 'fro'))
    reconstrucion_errors.append(recon_error)
        
    img1=transformed_X[0].reshape(16,16)
    img2=transformed_X[1].reshape(16,16)

    plt.imshow(img1,cmap='gray')
    plt.savefig('image1_%s.png'%p)
    plt.show()
    plt.imshow(img2,cmap='gray')
    plt.savefig('image2_%s.png'%p)
    plt.show()

plt.plot(P,reconstrucion_errors)
plt.xlabel('Value of P')
plt.ylabel('Total Reconstruction Error')
plt.savefig('recon_error.png')
plt.show()    