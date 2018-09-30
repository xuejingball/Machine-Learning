package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author Daniel Cohen dcohen@gatech.edu
 * @version 1.0
 */
public class TwoColorsTest {
    /** The number of colors */
    private static final int k = 2;
    /** The N value */
    private static final int N = 100*k;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, k+1);

        //int[] iterations = new int[]{ 20, 40, 60, 100, 120, 140, 200, 300, 400, 500, 600, 800, 1000};
        int[] iterations = new int[]{ 200 };
        double start, end, trainingTime;
        for (int i = 0; i < iterations.length; i ++) {
            System.out.println(iterations[i] + " iterations:");
            EvaluationFunction ef = new TwoColorsEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations[i]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            trainingTime = (end - start) / Math.pow(10,9);
            System.out.println(ef.value(rhc.getOptimal()));
            System.out.println("RHC training time: " + trainingTime + " seconds");

            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, iterations[i]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            trainingTime = (end - start) / Math.pow(10,9);
            System.out.println(ef.value(sa.getOptimal()));
            System.out.println("SA training time: " + trainingTime + " seconds");

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, iterations[i]);
            start = System.nanoTime();
            fit.train();
            end = System.nanoTime();
            trainingTime = (end - start) / Math.pow(10,9);
            System.out.println(ef.value(ga.getOptimal()));
            System.out.println("GA training time: " + trainingTime + " seconds");

            MIMIC mimic = new MIMIC(50, 10, pop);
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
