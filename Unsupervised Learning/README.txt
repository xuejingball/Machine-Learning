Reference https://github.com/danielcy715/CS7641-Machine-Learning/tree/master/Assignment3
1. My codes was written in python and utilize the scikit-learn package. 
2. This assignment was divided into three parts, every part you can find the corresponding code and support files.

For part 1, please refer to 3-5
3. This part was mainly about using two clustering algorithms. 
4. in the main code folder, to run clustering algorithm, you could open .py file in any python IDE. The name of the file should be dataset + algorithm name, such as adult_EM or ecoli_KMeans.
5. The plots of SSE or log likelihood were directly plotted in the console, which could by copied and saved. The plots of clustering visualization was also plotted directly in the console too.
6. Two csv files should be generated automatically after clustering code finishes. 
dataset_cluster_em: generated from EM
dataset_cluster_kmeans: generated from KMeans
7. Some other written data, like the correctly labeled cluster in percentage was calculated directly in Excel.

For part 2, please refer 8-13
8. This part was mainly about dimensionality reduction + clustering.
9. The dimensionality reduction was done in separate folders, named after the algorithm names, such as PCA, ICA, RP and RF.
10. The code was provided in the main code folder. You can run the .py file with the corresponding algorithm name. For example, PCA.py would data preprocess, run the component reduction analysis and at the end, generate the properly reduced new dataset. 
11. For each algorithm, the following files should be generated to be furthur organized and made plots for both datasets.
dataset scree.csv: metrics for each algorithm, such as eigenvalue for PCA and the absolute mean kurtosis for ICA.
dataset svm/dt validate: supervised learning algorithms validate the reduced dataset by testing accuracy.
dataset dim red: dimensionality reduced new dataset
some other information mentioned in the analysis report was likely to be directly printed in console, and copied to the report.
12. After dimensionality reduction, the clustering using two algorithms were separately finished in the folder of dimensionality reduced clustering. In that folder, you can run the .py files for each reduced csv file generated from previous step. The .py files were named like dataset_dr_algorithm name, such as adult_dr_em, means use reduced adult dataset and apply em.
13. The plots of SSE or log likelihood were directly plotted in the console, which could by copied and saved. The plots of clustering visualization was also plotted directly in the console too.

For part 3, please refer 14-16
14. Neural Network was applied on the original or reduced dataset by running NN.py. The data read-in process for the orginal dataset and reduced datasets were slightly different. So the NN.py file in the main code folder was for the original dataset, while the NN.py file in the 
subfolder of dimensionality reduced clustering was for DR reduced datasets.
15. Similar to the NN.py, the Neural Network was again applied to the clustering result froom the original and reduced datasets. The python code is named NN_cluster. The one in the main code folder was for the original clusterd dataset and the other one in the dimensionality reduced clustering folder was for the reduced clustered datasets.
16. Any Neural Network accuracy and training time would directly get printed in the console. But at the same time, two csv files would also be generated for the data records.
NN accuracy: stores the accuracy from original/reduced dataset + NN
NN cluster accuracy: stores the accuracy from original/reduced dataset + clustering + NN

17. The original dataset was the same as the ones used in Assignment 1 and Assignment 2, in the format of .arff. The data preprocessing was finished by a file called data_preprocess.py.
18. Besides the mentioned .py files, another helper.py and clustertesters were necessary for running some of the python files. 