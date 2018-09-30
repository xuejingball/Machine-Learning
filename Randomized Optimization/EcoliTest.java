package opt.test;

import dist.*;
import func.nn.activation.HyperbolicTangentSigmoid;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class EcoliTest {
    private static Instance[] instances = initializeInstances(268,"src/opt/test/Ecoli-train.txt");
    private static Instance[] testset = initializeInstances(68, "src/opt/test/Ecoli-test.txt");

    private static int inputLayer = 7, hiddenLayer = 8, outputLayer = 8, trainingIterations = 1400;
    //private static double learningRate = 0.2;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E15, .05, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 100, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double[] predicted = new double[8];
            double[] actual =  new double[8];
            String p, a;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                p = instances[j].getLabel().toString();
                a = networks[i].getOutputValues().toString();
                int begin = 0, over = 0, k = 0;
                while (over < p.length()){
                    if (p.charAt(over) == ','){
                        predicted[k] = Double.parseDouble(p.substring(begin, over));
                        actual[k] = Double.parseDouble(a.substring(begin, over));
                        k ++;
                        begin = over + 2;
                    }else if (over == p.length() - 1){
                        predicted[k] = Double.parseDouble(p.substring(begin));
                        actual[k] = Double.parseDouble(a.substring(begin));
                    }
                    over ++;
                }
                double trash = measureError(predicted, actual) ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nTRAIN Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            //calculate test error
            double starttest = System.nanoTime(), endtest, trainingTimetest, testingTimetest, correcttest = 0, incorrecttest = 0;
            double[] predictedtest = new double[8];
            double[] actualtest =  new double[8];
            String ptest, atest;
            starttest = System.nanoTime();
            for(int h = 0; h < testset.length; h++) {
                networks[i].setInputValues(testset[h].getData());
                networks[i].run();

                ptest = testset[h].getLabel().toString();
                atest = networks[i].getOutputValues().toString();
                int begin = 0, over = 0, k = 0;
                while (over < ptest.length()){
                    if (ptest.charAt(over) == ','){
                        predictedtest[k] = Double.parseDouble(ptest.substring(begin, over));
                        actualtest[k] = Double.parseDouble(atest.substring(begin, over));
                        k ++;
                        begin = over + 2;
                    }else if (over == ptest.length() - 1){
                        predictedtest[k] = Double.parseDouble(ptest.substring(begin));
                        actualtest[k] = Double.parseDouble(atest.substring(begin));
                    }
                    over ++;
                }
                double trashtest = measureError(predictedtest, actualtest) ? correcttest++ : incorrecttest++;
            }
            endtest = System.nanoTime();
            testingTimetest = endtest - starttest;
            testingTimetest /= Math.pow(10,9);

            results +=  "\nTEST Results for " + oaNames[i] + ": \nCorrectly classified " + correcttest + " instances." +
                    "\nIncorrectly classified " + incorrecttest + " instances.\nPercent correctly classified: "
                    + df.format(correcttest/(correcttest+incorrecttest)*100) + "%\n"
                    + "Testing time: " + df.format(testingTimetest) + " seconds\n";
        }

        System.out.println(results);
    }

    private static boolean measureError(double[] predicted, double actual[]){
        int index1 = 0;
        for (int n = 0; n < predicted.length; n++){
            if (predicted[n] > 0){
                index1 = n;
                break;
            }
        }
        int index2 = 0;
        double max = actual[0];
        for(int m = 1; m < actual.length; m++){
            if(actual[m] > max){
                max = actual[m];
                index2 = m;
            }
        }
        return index1==index2;
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(network.getOutputValues()));

                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static class attribute
    {
        public double numbers[];
        public String category;
        public attribute()
        {
            numbers =  new double[7];
            category = null;
        }
    }

    private static Instance[] initializeInstances(int length, String path) {

        int instanceLength = length;
        attribute[] attrs = new attribute[instanceLength];
        for (int i = 0; i < instanceLength; i++)
            attrs[i] = new attribute();

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(path)));

            for(int i = 0; i < instanceLength; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                for(int j = 0; j < 7; j++)
                    attrs[i].numbers[j] = Double.parseDouble(scan.next());

                attrs[i].category = scan.next();
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[instanceLength];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attrs[i].numbers);
            // classifications into 8 classes range from 0 to 7
            switch(attrs[i].category){
                case "cp": instances[i].setLabel(new Instance(new double[]{1,0,0,0,0,0,0,0})); break;
                case "im": instances[i].setLabel(new Instance(new double[]{0,1,0,0,0,0,0,0})); break;
                case "pp": instances[i].setLabel(new Instance(new double[]{0,0,1,0,0,0,0,0})); break;
                case "imU": instances[i].setLabel(new Instance(new double[]{0,0,0,1,0,0,0,0})); break;
                case "om": instances[i].setLabel(new Instance(new double[]{0,0,0,0,1,0,0,0})); break;
                case "omL": instances[i].setLabel(new Instance(new double[]{0,0,0,0,0,1,0,0})); break;
                case "imL": instances[i].setLabel(new Instance(new double[]{0,0,0,0,0,0,1,0})); break;
                case "imS": instances[i].setLabel(new Instance(new double[]{0,0,0,0,0,0,0,1})); break;
                default: System.err.println("found label " + attrs[i].category + " doesn't belong to data set");
            }
        }

        return instances;
    }
}
