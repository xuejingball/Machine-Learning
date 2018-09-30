
from clustertesters import adult_KMeansTestCluster as kmtc
from data_preprocessing import getAdultX, getAdultY
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

if __name__ == "__main__":
    X = getAdultX()
    y = getAdultY()
    
    tester = kmtc.KMeansTestCluster(X, y, clusters=range(1,11), plot=True, targetcluster=2, stats=True)
    tester.run()


# plot clustering
    kmeans = KMeans(n_clusters=3, max_iter=500, init='k-means++')
    labels = kmeans.fit_predict(X)

    # View the results
    # Set the size of the plot
    plt.figure(figsize=(14,7))
     
    # Create a colormap
    colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])
    x1 = X.iloc[:,0]
    x2 =  X.iloc[:,1]
    plt.scatter(x=x1,y=x2, c=colormap[labels], s=40)
    plt.title('ecoli K Mean Classification')


