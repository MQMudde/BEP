import os
import numpy as np
import csv


def get_hill_estimator(ordered_data):
    """
    Function to calculate Hill estimator array given an ordered data
    sequence. Decreasing ordering is required.
    :param ordered_data: numpy array of ordered data for which the 1st moment (Hill estimator) is calculated.
    :return: numpy array of Hill estimator corresponding to all possible order statistics of the dataset.
    """
    logs = np.log(ordered_data)
    logs_cumsum = np.cumsum(logs[:-1])
    k_vector = np.arange(1, len(ordered_data))
    m1 = (1./k_vector)*logs_cumsum - logs[1:]
    return m1


def get_hill_estimator_one_value(ordered_data, k):
    """
    Function to calculate the Hill estimator for a specified order statistic k. Decreasing ordering is required.
    :param ordered_data: Decreasingly ordered sample
    :param k: from 1 up to and including len(ordered_data) - 1
    :return: float with the value of the Hill estimator
    """
    selected_logs = np.log(ordered_data[:k+1])
    return 1./k * sum(selected_logs[:-1]) - selected_logs[-1]



def hills_from_sample_to_file(sample_file):
    """
    Function which calculates the hill estimator for each sample in a sample file
    and saves the result in another file in the same directory.
    :param sample_file: the path of the file containing the samples
    :return: the path of the file containing the Hill estimates
    """
    sample_file, _ = os.path.splitext(sample_file)
    hills_file = sample_file + ' Hill.csv'
    if os.path.isfile(hills_file):
        print(f'{hills_file} already exists, no new Hill estimation')
    else:
        with open(sample_file, 'r', newline='') as samples, open(hills_file, 'w', newline='') as hills:
            reader = csv.reader(samples, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
            writer = csv.writer(hills, delimiter=' ')
            for sample in reader:
                ret = get_hill_estimator(sample)
                writer.writerow(ret)
    return hills_file
