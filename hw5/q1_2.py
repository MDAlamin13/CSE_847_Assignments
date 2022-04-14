import numpy as np
import random
from matplotlib import pyplot as plt
np.random.seed(7)

def find_best_centroid(point,centroids):
    dists=[]
    for c in centroids:
        dist=(np.linalg.norm(point-c))**2
        dists.append(dist)
    best=np.argmin(dists)
    return best

def calc_new_centroid(list):
    sum=np.zeros(list[0].shape)
    for point in list:
        sum+=point 
    new_c=sum/len(list)
    return new_c  

def kmeans(X,k,iteration):
    random_indexes =np.random.randint(X.shape[0],size=k) 
    centroids=X[random_indexes]
    new_centroids=centroids

    assigned_dict={}
    for itr in range(iteration):
        for i in range(k):
            assigned_dict[i]=[]
            assigned_dict[i].append(centroids[i])

        for row in X:
            best_c=find_best_centroid(row,centroids)
            assigned_dict[best_c].append(row)

        for c in range(k):
            new_c=calc_new_centroid(assigned_dict[c])
            new_centroids[c]=new_c
        
        converge=True
        for c in range(k):
            diff=(np.linalg.norm(centroids[c]-new_centroids[c]))**2
            if(diff!=0):
                converge=False
                break

        if(converge==True):
            return assigned_dict
        else:
            centroids=new_centroids   

    return assigned_dict              


def plot(assignments):
    sections=[]
    for k in assignments:
        v=assignments[k]
        x=[]
        y=[]
        for p in v:   
            x.append(p[0])
            y.append(p[1])   
        sections.append([x,y])

    colors=['r','b','y']
    for i in range(len(sections)):

        plt.scatter(sections[i][0],sections[i][1],c=colors[i])
    plt.savefig('q1_kmean.png')
    plt.show()        


X=np.random.randint(30, size=(100,2))
assignments=kmeans(X,3,100)
plot(assignments)
