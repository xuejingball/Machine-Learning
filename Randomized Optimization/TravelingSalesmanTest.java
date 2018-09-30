package opt.test;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;

    private static DecimalFormat df = new DecimalFormat("0.000");
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        int[] iterations = new int[]{ 200, 400, 800, 1000, 2000,4000,10000,16000,20000,24000,40000,80000 };
        //int[] iterations = new int[]{ 200000 };
        double start, end, trainingTime;
        for (int i = 0; i < iterations.length; i ++) {
            System.out.println(iterations[i] + " iterations:");
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations[i]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            trainingTime = (end - start) / Math.pow(10,9);
            System.out.println(ef.value(rhc.getOptimal()));
            System.out.println("RHC training time: " + df.format(trainingTime) + " seconds");

            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            fit = new FixedIterationTrainer(sa, iterations[i]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            trainingTime = (end - start) / Math.pow(10,9);
            System.out.println(ef.value(sa.getOptimal()));
            System.out.println("SA training time: " + df.format(trainingTime) + " seconds");

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 50, gap);
            fit = new FixedIterationTrainer(ga, iterations[i]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            trainingTime = (end - start) / Math.pow(10,9);
            System.out.println(ef.value(ga.getOptimal()));
            System.out.println("GA training time: " + df.format(trainingTime) + " seconds");

            // for mimic we use a sort encoding
            ef = new TravelingSalesmanSortEvaluationFunction(points);
            int[] ranges = new int[N];
            Arrays.fill(ranges, N);
            odd = new DiscreteUniformDistribution(ranges);
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            MIMIC mimic = new MIMIC(200, 100, pop);
            fit = new FixedIterationTrainer(mimic, iterations[i]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            trainingTime = (end - start) / Math.pow(10,9);
            System.out.println(ef.value(mimic.getOptimal()));
            System.out.println("MIMIC training time: " + trainingTime + " seconds");
            System.out.println();
        }
    }
}
