import Sampling
import Hill
import Measuring
import Manipulation
import time
import itertools

def get_results_from_scratch(sample_size, nr_samples, sample_type, xi, rejection_prob, save_hill_estimation = False):
    start_time = time.time()
    # Sampling
    print(f'Started sampling {nr_samples} samples of {sample_size} {sample_type}')
    sample_file = Sampling.Sampler(xi, sample_type).sample_to_file(sample_size, nr_samples)
    end_sampling = time.time()
    print(f'Finished sampling in {round(end_sampling - start_time, 2)} seconds')
    if save_hill_estimation:
        # Hill estimation
        print('Started writing hill estimator to file')
        hills_file = Hill.hills_from_sample_to_file(sample_file)
        end_hill = time.time()
        print(f'Finished writing hill estimator to file in {round(end_hill - end_sampling,2)} seconds')
    # Measurement execution
    print('Started measuring')
    start_measuring = time.time()
    Measuring.Measuring(sample_file, rejection_prob).writing_results_to_files()
    end_measuring = time.time()
    print(f'Finished measuring in {round(end_measuring - start_measuring,2)} seconds')
    print(f'Total run time {round(end_measuring - start_time,2)} seconds \n which is {round((end_measuring - start_time)/60)} minutes')
    return sample_file


def get_results_from_noise(sample_file, noise_type, rejection_prob, save_hill_estimation=False, **kwargs):
    start_time = time.time()
    # manipulation
    manipulator = Manipulation.SampleManipulation(sample_file, noise_type, kwargs)
    print(f'Started adding {noise_type} noise')
    sample_with_noise_file = manipulator.add_noise_from_file()
    end_adding_noise = time.time()
    print(f'Finished adding noise in {round(end_adding_noise - start_time, 2)} seconds')
    if save_hill_estimation:
        # Hill estimation
        print('Started writing Hill estimator to file')
        hills_file = Hill.hills_from_sample_to_file(sample_with_noise_file)
        end_hill = time.time()
        print(f'Finished writing hill estimator to file in {round(end_hill - end_adding_noise, 2)} seconds')
    # Measurement execution
    start_measuring = time.time()
    print('Started measuring')
    Measuring.Measuring(sample_with_noise_file, rejection_prob).writing_results_to_files()
    end_measuring = time.time()
    print(f'Finished measuring in {round(end_measuring - start_measuring, 2)} seconds')
    print(f'Total run time {round(end_measuring - start_time, 2)} seconds '
          f'\n which is {round((end_measuring - start_time) / 60)} minutes')


if __name__ == "__main__":
    """"
    Here the values for the simulation can be set. 
    Preferential attachment should only be ran with xi 0.5, otherwise the simulation is invalid. 
    If save_hill_estimates is True, all the resulting Hill estimates are written to files. Note that this can create substantial file sizes.
    If sample files or files with samples with noise added to them already exist, they are not written again.
    To resample, delete the sample files first. 
    Measurement files are rewritten when rerunning the code.
    """
    xis = [0.5]#[0.5, 0.6, 0.7, 0.8]
    samplesizes = [10_000]#[100, 200, 500, 1000, 2000, 5000, 10000]
    nrsamples = 10_000
    sampletypes = ['Zipf']#['RoundedPareto', 'MixedPoisson', 'Zipf']  #["PreferentialAttachment"]
    rejection_prob_alpha = 0.1
    save_hill_estimates = False

    noisetypes = ['Pareto', 'Uniform', 'Gaussian']
    # dictionary with all the parameters for each noise type that will be used in the simulation.
    noisetype_param_dict = {
        'Pareto': [['xi'], [[0.5], [0.6], [0.7], [0.8]]],
        'Uniform': [['loc', 'scale'], [[0, 1], [-1, 1], [-0.5, 1], [-0.5, 0.5], [0, 0.5], [-0.25, 0.5]]],
        'Gaussian': [['lower', 'upper'], [[-0.5, 0.5], [0, 1], [-1, 0]]]}

    # Loop running over all selected simulation values. All noise types will be added to each simulation value.
    for samplesize, sampletype, xi in itertools.product(samplesizes, sampletypes, xis):
        samples_file = get_results_from_scratch(samplesize, nrsamples, sampletype, xi, rejection_prob_alpha, save_hill_estimation=save_hill_estimates)
        for noisetype in noisetypes:
            param_list = noisetype_param_dict[noisetype][0]
            param_list_values = noisetype_param_dict[noisetype][1]
            for param_values in param_list_values:
                param_dict = {param: param_value for param, param_value in zip(param_list, param_values)}
                get_results_from_noise(samples_file, noisetype, rejection_prob_alpha, **param_dict, save_hill_estimation=save_hill_estimates)






