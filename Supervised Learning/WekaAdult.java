import java.io.*;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.core.converters.LibSVMLoader;
import weka.core.pmml.jaxbbindings.Output;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemovePercentage;

public class WekaAdult {
    public static Evaluation classify(Classifier model,
                                      Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);
        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);

        return evaluation;
    }

    public static String crossValidation(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Debug.Random(1));

        String result = "crossValidation error rates: " + eval.errorRate() + "\n";

        System.out.println("crossValidation error rates: " + eval.errorRate());
        return result;
    }

    public static String training(Classifier model, Instances train, Instances test) throws Exception {
        Evaluation eval = classify(model, train, test);
        String result = "train error rates： " + eval.errorRate() + "\n";
        System.out.println("train error rates: " + eval.errorRate());
        return result;
    }

    public static String testing(Classifier model, Instances train, Instances test, double percentage) throws Exception {
        int trainSize = (int) Math.round(train.numInstances() * percentage);
        Instances trainDataSet = new Instances(train, 0, trainSize);
        Evaluation eval = classify(model, trainDataSet, test);
        String result = model.toString() + "\n" + eval.toSummaryString() + "\n";
        System.out.println("test error rates: " + eval.errorRate());
        return result;
    }

    private static void OutputStream(String data, String filename) {
        OutputStream os = null;
        try {
            os = new FileOutputStream(new File("src\\" + filename));
            os.write(data.getBytes(), 0, data.length());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                os.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static String calculate(Classifier model, Instances data) {
        // remove from the train dataset, using various percentages
        double[] percentages = {100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5};
        String errorRates = "";
        String result = "";

        for (int i = 0; i < percentages.length; i++) {
            double percentage = percentages[i] / 100;

            int trainSize = (int) Math.round(data.numInstances() * percentage);
            Instances trainDataSet = new Instances(data, 0, trainSize);

            System.out.println(percentages[i] + " percentage of data has been used.");

            try {
                //Evaluate the performance on training dataset

                Evaluation evaluationTrain = classify(model, trainDataSet, trainDataSet);
                System.out.println("train error rates: " + evaluationTrain.errorRate());

                //Evaluate the performance on testing dataset
                result = crossValidation(model, trainDataSet);

                errorRates += Double.toString(percentages[i]) + " train error rates: " + evaluationTrain.errorRate() + "\n" +
                        Double.toString(percentages[i]) + " " + result + "\n";


            } catch (Exception e) {
                System.err.println("data error");
            }
        }
        return errorRates;
    }

    public static void main(String[] args) throws Exception {
        //read in train data source
        ConverterUtils.DataSource source =
                new ConverterUtils.DataSource("src\\adult.arff");

        //set train data index
        Instances dataSource = source.getDataSet();
        dataSource.randomize(new Debug.Random(123));
        if (dataSource.classIndex() == -1) {
            dataSource.setClassIndex(dataSource.numAttributes() - 1);
        }

        int seed = (int) Math.random() * 50 + 1;
        int folds = 5;
        // randomize data
        Debug.Random rand = new Debug.Random(seed);
        Instances randData = new Instances(dataSource);
        randData.randomize(rand);
        if (randData.classAttribute().isNominal())
            randData.stratify(folds);
        Instances target = randData.trainCV(folds, 0);
        Instances data = randData.testCV(folds, 0);

        data.randomize(new Debug.Random(789));

        /*ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("Adult-train.arff"));
        saver.writeBatch();

        saver.setInstances(target);
        saver.setFile(new File("Adult-test.arff"));
        saver.writeBatch();*/

        Classifier[] models = {
                new J48(), // a decision tree with pruning
                new MultilayerPerceptron(), //Neural Networks
                new AdaBoostM1(),//Boosting
                new IBk(), //K-Nearest Neighbors
                new SMO() //Support Vector Machine
        };
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //Tune the hyperparameters for J48 model
        /*
        // optimize confidence factor c
        float[] conf = {0.01f, 0.05f, 0.1f, 0.25f, 0.3f, 0.4f, 0.5f};
        String cv_J48 = "";
        String filename = "J48-traning result of confidence factor.txt";
        for(int c = 0; c < conf.length; c++){
            ((J48)models[0]).setConfidenceFactor(conf[c]);
             cv_J48 += crossValidation(models[0], data);
             cv_J48 += training(models[0], data, data);
        }
        OutputStream(cv_J48, filename);*/

        /*
        // optimize the minimum number of objects m
        int[] numNodes = {2, 5, 10, 15, 20, 25, 50, 100};
        String cv_J48_nodes = "";
        String filename1 = "J48-traning result of nodes.txt";
        for(int n = 0; n < numNodes.length; n++){
            ((J48)models[0]).setConfidenceFactor(0.1f);
            ((J48)models[0]).setMinNumObj(numNodes[n]);
            cv_J48_nodes += crossValidation(models[0], data);
            cv_J48_nodes += training(models[0], data, data);
        }
        OutputStream(cv_J48_nodes, filename1);*/

        //Calculate training and cv errors using model J48
        /*((J48)models[0]).setConfidenceFactor(0.1f);
        String res = calculate(models[0], data);
        OutputStream(res, "J48-traing result-Adult.txt");*/

        //---------------------------------------------------------------------

        //Tune the hyperparameters for model IBk
        /*
        // optimize the number of nearest neighbors KNN
        String filename = "KNN-traning result of K.txt";
        String KNN_K ="";
        for (int k = 1; k <= data.numInstances(); k = k*2){
            ((IBk)models[3]).setKNN(k);
            KNN_K += crossValidation(models[3], data);
            KNN_K += training(models[3], data, data);
        }
        OutputStream(KNN_K, filename);*/

        //Calculate using model IBk
        /*((IBk)models[3]).setKNN(64);
        String KNN = calculate(models[3], data);
        OutputStream(KNN, "KNN-traing resul-Adult.txt");*/

        //-------------------------------------------------------------------------------

        //Tune the hyperparameters for Multilayererceptron
        //By experience equation: hidden layer = 1; the nodes in that layer = (sqrt(input nodes+output nodes)) + 1-10
        /*
        // optimize the number of nodes in a single hidden layer N
        String filename = "NN-training result of hidden layers.txt";
        String NN_hl= "";
        for(int hl = 0; hl<=14; hl+=1){
            ((MultilayerPerceptron)models[1]).setHiddenLayers(Integer.toString(hl));
            NN_hl += crossValidation(models[1], data);
            NN_hl += training(models[1], data, data);
        }
        OutputStream(NN_hl, filename);*/

        /*
        // optimize the learning rate r
        String filename = "NN-training result of learning rates.txt";
        String NN_lr = "";
        for(double lr = 0.1; lr<=0.5; lr=lr+0.1){
            ((MultilayerPerceptron)models[1]).setHiddenLayers("0");
            ((MultilayerPerceptron)models[1]).setLearningRate(lr);
            NN_lr += crossValidation(models[1], data);
            NN_lr += training(models[1],data, data);
        }
        OutputStream(NN_lr, filename);*/


        /*
        // optimize the number of iterations
        String filename = "NN-training result of iterations.txt";
        String NN_iterations = "";
        for(int times = 1; times <= 1024 ; times*=2){
            ((MultilayerPerceptron)models[1]).setHiddenLayers("0");
            ((MultilayerPerceptron)models[1]).setLearningRate(0.2);
            ((MultilayerPerceptron)models[1]).setTrainingTime(times);
            NN_iterations += crossValidation(models[1], data);
            NN_iterations += training(models[1],data, data);
        }
        OutputStream(NN_iterations, filename);*/

        //Calculate using model MultilayerPerceptron
        /*((MultilayerPerceptron)models[1]).setLearningRate(0.2);
        ((MultilayerPerceptron)models[1]).setHiddenLayers("0");
        ((MultilayerPerceptron)models[1]).setTrainingTime(512);
        String NN = calculate(models[1], data);
        OutputStream(NN, "NeuralNetworks-traing result-Adult.txt");*/

        //-------------------------------------------------------------
        //Tune the hyperparameters for AdaBoostM1
        /*
        // optimize the number of iterations
        String filename = "Boosting-training result of iterations.txt";
        String B_iterations = "";
        for(int times = 1; times <= 1024 ; times*=2){
            Classifier dt = new J48();
            ((J48)dt).setConfidenceFactor(0.1f);
            ((AdaBoostM1)models[2]).setClassifier(dt);
            ((AdaBoostM1)models[2]).setNumIterations(times);
            B_iterations += crossValidation(models[2], data);
            B_iterations += training(models[2],data, data);
        }
        OutputStream(B_iterations, filename);*/

        //Calculate using model AdaBoostM1(J48)(Boosted J48)
        /*Classifier dt = new J48();
        ((J48) dt).setConfidenceFactor(0.1f);
        ((AdaBoostM1) models[2]).setClassifier(dt);
        ((AdaBoostM1) models[2]).setNumIterations(32);
        String boostedDT = calculate(models[2], data);
        OutputStream(boostedDT, "BoostedDT-traing result-Adult2.txt");*/

        //------------------------------------------------------------------

        //Tune the hyperparameters of SMO model
        /*
        // optimize the complexity paramter c for poly kernel
        String filename = "SMO-training result of complexity parameter.txt";
        double[] C = {0.01, 0.1, 1, 5, 6, 7, 8, 9};
        String SMO_c= "";
        for(int c = 0; c<C.length; c++){
            ((SMO)models[4]).setKernel(new PolyKernel());
            ((SMO)models[4]).setC(C[c]);
            SMO_c += crossValidation(models[4], data);
            SMO_c += training(models[4], data, data);
        }
        OutputStream(SMO_c, filename);*/

        /*
        // optimize the complexity paramter c for RBF kernel
        String filename = "SMO_RBF-training result of complexity parameter.txt";
        double[] C = {0.01, 0.1, 1, 10, 20, 60, 100, 500, 1000};
        String SMO_RBF_c= "";
        for(int c = 0; c<C.length; c++){
            ((SMO)models[4]).setKernel(new RBFKernel());
            ((SMO)models[4]).setC(C[c]);
            SMO_RBF_c += crossValidation(models[4], data);
            SMO_RBF_c += training(models[4], data, data);
        }
        OutputStream(SMO_RBF_c, filename);*/

        /*
        // optimize the gamma g for RBF kernel
        String filename = "SMO_RBF-training result of gamma.txt";
        String SMO_RBF_gamma= "";
        for(double Y=0; Y<=-1; Y++){
            double YBase = 10;
            double g = Math.pow(YBase, Y);
            RBFKernel RBF = new RBFKernel();
            RBF.setGamma(g);
            ((SMO)models[4]).setKernel(RBF);
            ((SMO)models[4]).setC(500);
            SMO_RBF_gamma += crossValidation(models[4], data);
            SMO_RBF_gamma += training(models[4], data, data);
        }
        OutputStream(SMO_RBF_gamma, filename);*/

        //Calculate using model SMO(Support Vector Machine) with PolyKernel
        /*((SMO)models[4]).setKernel(new PolyKernel());
        ((SMO)models[4]).setC(8);
        String SVM_polyKernel = calculate(models[4], data);
        OutputStream(SVM_polyKernel, "SVM_PolyKernel-training result-Adult.txt");*/

       /* RBFKernel RBF = new RBFKernel();
        RBF.setGamma(Math.pow(10, -2));
        ((SMO)models[4]).setKernel(RBF);
        ((SMO)models[4]).setC(500);
        String SVM_RBF = calculate(models[4], data);
        OutputStream(SVM_RBF, "SVM_RBF-training result-Adult.txt");*/

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //Test all algorithms using test data
         /*((J48)models[0]).setConfidenceFactor(0.1f);
        String res = testing(models[0], data, target, 1);
        OutputStream(res, "J48-test result-Adult.txt");*/

        /*((IBk)models[3]).setKNN(8);
        String result = testing(models[3], data, target, 0.5);
        OutputStream(result, "KNN_test result-Adult.txt");*/

        /*((MultilayerPerceptron)models[1]).setLearningRate(0.0);
        ((MultilayerPerceptron)models[1]).setHiddenLayers("0");
        ((MultilayerPerceptron)models[1]).setTrainingTime(512);
        String result2 = testing(models[1], data, target, 0.5);
        OutputStream(result2, "NN_test result-Adult.txt");*/

        /*Classifier dt = new J48();
        ((J48) dt).setConfidenceFactor(0.1f);
        ((AdaBoostM1) models[2]).setClassifier(dt);
        ((AdaBoostM1) models[2]).setNumIterations(32);
        String result3 = testing(models[2], data, target, 1);
        OutputStream(result3, "BoostedDT-test result-Adult.txt");*/

        /*((SMO)models[4]).setKernel(new PolyKernel());
        ((SMO)models[4]).setC(8);
        String result4 = testing(models[4], data, target, 1);
        OutputStream(result4, "SVM_PolyKernel-test result-Adult.txt");*/

       /* RBFKernel RBF = new RBFKernel();
        RBF.setGamma(Math.pow(10, -2));
        ((SMO)models[4]).setKernel(RBF);
        ((SMO)models[4]).setC(500);
        String result5 = testing(models[4], data, target, 1);
        OutputStream(result5, "SVM_RBF-test result-Adult.txt");*/
    }
}