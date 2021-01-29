import csv
import scipy.stats
import math
import os
import networkx
from Distribution import Distribution


# The Sampler class is used to sample random variables from certain distributions.
# Objects of this class have a dist object inside.
# This object has a callable function rvs which returns a specified size random sample.
# sample_to_file saves sorted samples to a file in the directory with relative path ../Data/
class Sampler:

    def __init__(self, xi, sample_type):

        def pareto_dist():
            return Distribution(scipy.stats.pareto(1 / self.xi))

        def zipf_dist():
            return Distribution(scipy.stats.zipf(1 / self.xi + 1))

        def rounded_pareto_dist():
            return RoundedPareto(self.xi)

        def mixed_poisson_dist():
            return MixedPoisson(self.xi)

        def preferential_attachment():
            return PreferentialAttachment(1)

        self.sampleType = sample_type
        self.xi = xi
        self.sampleTypeDict = {'Pareto': pareto_dist, 'Zipf': zipf_dist,
                               'RoundedPareto': rounded_pareto_dist, 'MixedPoisson': mixed_poisson_dist,
                               'PreferentialAttachment': preferential_attachment}
        self.dist = self.sampleTypeDict[self.sampleType]()

    def generate_sample(self, sample_size):
        """
        Generate a random sample and returns it in DECREASING order.
        :param sample_size: size of the sample
        :return: random sample sorted in decreasing order
        """
        return sorted(self.dist.rvs(sample_size), reverse=True)

    def sample_to_file(self, sample_size, nr_samples):
        """
        Function to generate samples in an appropriately named file.
        :param sample_size: size of the samples
        :param nr_samples: number of samples
        :return: the file location of the sample (relative filepath)
        """
        name = f'{self.sampleType} {str(self.xi)} xi {str(sample_size)} n {str(nr_samples)} samples'
        try:
            os.mkdir(f'../Data/{name}')
        except OSError:
            print(f'The map ../Data/{name} already exists, continued.')
        filename = f'../Data/{name}/{name} None.csv'
        if os.path.isfile(filename):
            print(f'{filename} already exists, not sampled again.')
        else:
            with open(filename, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=' ')
                for _ in range(nr_samples):
                    writer.writerow(self.generate_sample(sample_size))
        return filename


# Minimal object which has a rvs function. Is used in conjunction with the Sampler class.
class RoundedPareto:

    def __init__(self, xi):
        self.dist = Distribution(scipy.stats.pareto(1/xi))

    def rvs(self, size=1):
        return [math.floor(pareto) for pareto in self.dist.rvs(size)]


# Minimal object which has a rvs function. Is used in conjunction with the Sampler class.
class MixedPoisson:
    def __init__(self, xi):
        self.distpareto = Distribution(scipy.stats.pareto(1/xi))
        self.poisson = scipy.stats.poisson

    def rvs(self, size=1):
        pareto_sample = self.distpareto.rvs(size)
        return [self.poisson.rvs(pareto, loc=1) for pareto in pareto_sample]


# Minimal object which has a rvs function. Is used in conjunction with the Sampler class.
class PreferentialAttachment:

    def __init__(self, m):
        self.generator = networkx.generators.random_graphs.barabasi_albert_graph
        self.m = m

    def rvs(self, size=1):
        graph = self.generator(size, m=self.m)
        return [degree for (_, degree) in networkx.classes.function.degree(graph)]
