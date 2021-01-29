import csv
import numpy as np
import scipy.stats
import os
import copy
import Hill


# The measuring class is used to determine the results from a file with samples.
# Initializing of the results which will be measured is done in __init__.
# In writing_results_to_files the sample is read out, Hill estimates are calculated and results are stored.
# When the sample file is read completely, the results are stored in an appropriately named files in the same directory.
class Measuring:

    confidence_interval_types = ['Wald', 'Score', 'LR', 'BCLR']

    def __init__(self, sample_file_path, alpha):
        # remove extension
        sample_file_path_no_ext, _ = os.path.splitext(sample_file_path)
        # Useful file names
        self.sample_file_path = sample_file_path
        self.sample_file_path_no_ext = sample_file_path_no_ext
        # Gathering parameters from file name
        _, sample_file_name = os.path.split(sample_file_path)
        self.sample_size = int(sample_file_name.split()[3])
        self.nr_samples = int(sample_file_name.split()[5])
        self.gamma = float(sample_file_name.split()[1])
        # Initialize variance and distance array
        self.measurement_variance_results, self.measurement_distance_results = [], []
        # Initialize confidence intervals
        # start zero lists for results, stored in a dict
        zero_lists = [(self.sample_size - 1) * [0] for _ in range(len(Measuring.confidence_interval_types))]
        self.confidence_interval_avg_sizes = {type: start_list for type, start_list in
                                              zip(Measuring.confidence_interval_types, zero_lists)}
        self.rejection_probabilities = copy.deepcopy(self.confidence_interval_avg_sizes)
        # Make confidence interval calculations so they can be reused
        z_half_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
        k_vec_sqrt = np.array([1 / np.sqrt(k + 1) for k in range(self.sample_size - 1)])
        k_vec = np.array([1 / (k + 1) for k in range(self.sample_size - 1)])
        k_vec_pow_3_over_2 = k_vec * k_vec_sqrt

        def q_k_LR(z):
            return z + 1/3 * z**2 * k_vec_sqrt + 1/36 * z**3 * k_vec - 1/270 * z**4 * k_vec_pow_3_over_2

        def q_k_BCLR(z):
            return z + 1/3 * z**2 * k_vec_sqrt + (1/36 * z**3 + 1/12 * z) * k_vec \
                   + (-1/270 * z**4 + 1/18 * z**2) * k_vec_pow_3_over_2

        # Resulting arrays are stored
        self.boundary_dict = \
            {'Wald': (1 - k_vec_sqrt * z_half_alpha, 1 + k_vec_sqrt * z_half_alpha),
             'Score': (1 / (1 + k_vec_sqrt * z_half_alpha), 1 / (1 - k_vec_sqrt * z_half_alpha)),
             'LR': (1 / (q_k_LR(z_half_alpha) * k_vec_sqrt + 1), 1 / (q_k_LR(-z_half_alpha) * k_vec_sqrt + 1)),
             'BCLR': (1 / (q_k_BCLR(z_half_alpha) * k_vec_sqrt + 1), 1 / (q_k_BCLR(-z_half_alpha) * k_vec_sqrt + 1))}
        # Initialize MSE array
        self.measurement_mse_results = np.zeros(self.sample_size - 1)

    def measure_variance(self, hills):
        self.measurement_variance_results.append(np.var(hills))

    def measure_distance(self, hills1, hills2, p):
        # euclidean norm if p=2
        self.measurement_distance_results.append(np.linalg.norm(np.array(hills1) - np.array(hills2), ord=p))

    def measure_confidence_intervals(self, hills):
        for type in Measuring.confidence_interval_types:
            intervals = zip(self.boundary_dict[type][0] * hills, self.boundary_dict[type][1] * hills)
            for index, (lower, upper) in enumerate(intervals):
                size = upper - lower
                self.confidence_interval_avg_sizes[type][index] += size
                if self.gamma < lower or self.gamma > upper:
                    self.rejection_probabilities[type][index] += 1

    def measure_mean_squared_error(self, hills):
        hills_array = np.array(hills)
        self.measurement_mse_results += (hills_array - self.gamma) * (hills_array - self.gamma)

    def writing_results_to_files(self):
        """
        Function which calculates all the performance measures for a supplied sample file
        :return: None
        """
        with open(self.sample_file_path, 'r', newline='') as samples:
            reader = csv.reader(samples, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
            for sample in reader:
                # Get Hill estimator
                hills_sample = Hill.get_hill_estimator(sample)
                # Measure variance
                self.measure_variance(hills_sample)
                # Measure distance
                if reader.line_num % 2 == 0:
                    sample_even = hills_sample
                    self.measure_distance(sample_even, sample_odd, 2)
                else:
                    sample_odd = hills_sample
                # Measure confidence intervals
                self.measure_confidence_intervals(hills_sample)
                # Measure MSE
                self.measure_mean_squared_error(hills_sample)
        # Write variance to file
        write_result_to_file(self.sample_file_path_no_ext + ' Variance.csv', self.measurement_variance_results)
        # Write distance to file
        write_result_to_file(self.sample_file_path_no_ext + ' Distance.csv', self.measurement_distance_results)
        # Write MSE to file
        write_result_to_file(self.sample_file_path_no_ext + ' MSE.csv', self.measurement_mse_results / self.nr_samples)
        # Confidence intervals
        # Divide occurences by the total samples to get averages/probabilities
        for type, occurences in self.confidence_interval_avg_sizes.items():
            self.confidence_interval_avg_sizes[type] = [occurence / self.nr_samples for occurence in occurences]
        for type, occurences in self.rejection_probabilities.items():
            self.rejection_probabilities[type] = [occurence / self.nr_samples for occurence in occurences]
        # Write confidence interval size to files
        for type, results in self.confidence_interval_avg_sizes.items():
            write_result_to_file(self.sample_file_path_no_ext + f' {type}Intervals.csv', results)
        # Write rejection probabilities to files
        for type, results in self.rejection_probabilities.items():
            write_result_to_file(self.sample_file_path_no_ext + f' {type}Rejection.csv', results)


def write_result_to_file(file_path, result):
    """
    Helper function to write results to files. Used in Measuring.writing_results_to_files
    :param file_path: path of the file to which the results will be written. Should have extension .csv
    :param result: list or array of numbers which will be stored in file_path
    :return: None
    """
    with open(file_path, 'w', newline='') as results_file:
        writer_results = csv.writer(results_file, delimiter=' ')
        writer_results.writerow(result)
