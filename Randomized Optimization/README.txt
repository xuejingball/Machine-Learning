1. My codes utilize the Abagail package. 
2. To run my codes for randomized optimization assignment, you open the Abagail package as a project in any JAVA IDEA, such as IntelliJ which is what I am using. 
3. Copy or move any JAVA file from the "java code" folder into the src/opt/test folder of the Abagail project. EcoliTest.java is for the neural network study in part 1 and the rest three is to replace the files with the same name in Abagail for part 2 of this assignment.

For part 1, please refer to 4-9
4. Copy or move TXT data files from the "supporting files" folder into the src/opt/test folder of the Abagail project, for the proper file read-in. Ecoli-train.txt is the data set for training, and Ecoli-test.txt is the data set for testing.
5. Now you can run the JAVA file by clicking “RUN” or directly run the main function in the java file.
6. In the output window of IntelliJ or any other IDE, the accuracy in percentages were printed after "percent correctly classified: ". The error rate in my report was equal to 1 - accuracy. 
7. Training and testing time was also printed in the output window after texts of "training time: " and "testing time: ".
8. For different number of iterations, I manually changed the number of trainingIterations at the top in the java code. 
9. For each iteration number, I clicked "run" and ran five times to take average for the final result.
10. Collected data was summarized and organized in Excel. All files were provided in the "supporting files" folder. Plots and graphs were made correspondingly. 
Ecoli setting: Figure 1, and Figure 5.
RHC-NN: Figure 2
SA-NN: Figure 3
GA-NN: Figure 4
NN-learning curve: data from assignment 1

For part 2, please refer 11-14
11. open any java file from the "supporting files" folder in the Abagail project. Run it in Intellij or any java IDE. 
12. All the results should be printed in the output window. such as:
TRAIN Results for RHC/SA/GA/MIMIC: 
Correctly classified xxx instances.
Incorrectly classified xxx instances.
Percent correctly classified: xxx%
Training time: xxx seconds
Testing time: xxx seconds

TEST Results for RHC: 
Correctly classified xxx instances.
Incorrectly classified xxx instances.
Percent correctly classified: xxx %
Testing time: xxx seconds

13. For each iteration number, I clicked "run" and ran five times to take average for the final result.
14. Collected data was summarized and organized in Excel. All files were provided in the "supporting files" folder. Plots and graphs were made correspondingly. 
travelingsalesman: Figure 7
8queens: Figure 8
twocolors: Figure 9