

from clustertesters import adult_ExpectationMaximizationTestCluster as emtc
from data_preprocessing import getAdultX, getAdultY
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    X = getAdultX()
    y = getAdultY()

    tester = emtc.ExpectationMaximizationTestCluster(X, y, clusters=range(1,11), plot=True, targetcluster=2, stats=True)
    tester.run()

# plot clustering
    gmm = GMM(covariance_type = 'diag', n_components=3)
    model = gmm.fit(X)
    labels = model.predict(X)
    # View the results
    # Set the size of the plot
    plt.figure(figsize=(14,7))
     
    # Create a colormap
    colormap = np.array(['red', 'lime', 'black', 'blue', 'yellow'])
    x1 = X.iloc[:,0]
    x2 =  X.iloc[:,1]
    plt.scatter(x=x1,y=x2, c=colormap[labels], s=40)
    plt.title('adult EM Classification')
