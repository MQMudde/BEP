import csv
import scipy.stats
import os
import numpy as np
from Distribution import Distribution
import itertools

# The SampleManipulation class is used to add noise to samples.
# Objects of this class have a dist object inside.
# This object has a callable function rvs which returns a specified size random noise sample.
# This in turn can be used to add noise to the original sample.
# sample_to_file saves sorted samples to a file in the directory with relative path ../Data/
class SampleManipulation:

    def __init__(self, sample_file, noise_type, dict_kwargs):

        def uniform_noise(loc=0, scale=1):
            return Distribution(scipy.stats.uniform(loc=loc, scale=scale))

        def gaussian_noise(lower=-0.5, upper=0.5):
            return Distribution(scipy.stats.truncnorm(a=lower, b=upper))

        noise_type_dict = {'Uniform': uniform_noise, 'Gaussian': gaussian_noise, 'Pareto': ParetoNoise}
        self.location_dependent_noise_types = ['Pareto']
        self.sample_file = sample_file
        self.noise_type = noise_type
        self.dist = noise_type_dict[noise_type](**dict_kwargs)
        self.dict_kwargs = dict_kwargs

    def add_noise(self, sample):
        """
        Function which adds noise to a specified sample. Sample should be in decreasing order
        :param sample: list of a sample in decreasing order
        :return: list of sample + noise, again in decreasing order
        """
        if self.noise_type in self.location_dependent_noise_types:
            sample_with_noise = self.dist.rvs(sample)
        else:
            noise_samples = self.dist.rvs(len(sample))
            sample_with_noise = sorted([value + noise for value, noise in zip(sample, noise_samples)], reverse=True)
        return sample_with_noise

    def add_noise_from_file(self):
        """
        Function which adds noise to the samples from the sample file originally specified in the initialization of the
        SampleManipulation object. The noisy sample is saved in another appropriately named file in the same directory.
        :return: the path of the noisy sample file (relative)
        """
        # remove 'None' from filename and add 'noise_type' and keyword arguments to get target file
        dict_string = ''
        for key, value in self.dict_kwargs.items():
            dict_string += str(key) + str(value)
        target_file = self.sample_file.rsplit(' ', 1)[0] + ' ' + self.noise_type + ' ' + dict_string + '.csv'
        if os.path.isfile(target_file):
            print(f'{target_file} already exists, no new noise added')
        else:
            with open(self.sample_file, 'r', newline='') as sample_file, open(target_file, 'w', newline='') as with_noise:
                sample_reader = csv.reader(sample_file, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
                noise_writer = csv.writer(with_noise, delimiter=' ')
                for sample in sample_reader:
                    sample_with_noise = self.add_noise(sample)
                    noise_writer.writerow(sample_with_noise)
        return target_file

# Used to construct the Pareto noise, which is truncated.
class TruncatedDistribution:
    # see https://timvieira.github.io/blog/post/2020/06/30/generating-truncated-random-variates/

    def __init__(self, d, a, b):
        assert np.all(a <= b), [a, b]
        self.d = d; self.a = a; self.b = b
        self.cdf_b = d.cdf(b)
        self.cdf_a = d.cdf(a)
        self.cdf_w = self.cdf_b - self.cdf_a
        self.rand = scipy.stats.uniform()

    def rvs(self, size=None):
        u = self.rand.rvs(size)
        return self.ppf(u)

    def ppf(self, u):
        return self.d.ppf(self.cdf_a + u * self.cdf_w)


class ParetoNoise:

    def __init__(self, xi=0.5):
        self.dist = scipy.stats.pareto(1/xi)

    def rvs(self, sample):
        noisy_sample = []
        groups = itertools.groupby(sample)
        grouped_sample = [(number, sum(1 for _ in group)) for number, group in groups]
        for number, multiplicity in grouped_sample:
            truncated_dist = TruncatedDistribution(self.dist, number, number+1)
            noise_rvs = sorted(truncated_dist.rvs(multiplicity), reverse=True)
            noisy_sample.extend(noise_rvs)
        return noisy_sample
