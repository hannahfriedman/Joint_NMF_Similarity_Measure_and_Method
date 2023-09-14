import matplotlib.pyplot as plt
import statistics as stats
import math as m
import random
import numpy as np
from copy  import deepcopy
from similarity import sim
from similarity import compute_chamfer_dist
from scipy.io import loadmat
import warnings 
warnings.filterwarnings("ignore")
plt.rc('text', usetex=True)


def listStats(numList):
    """takes in a list of numbers and returns [average, standard deviation, 95% confidence interval]"""
    avg = stats.mean(numList)
    std = stats.stdev(numList)
    sem = std / m.sqrt(len(numList))
    z = 1.96 # 95% ci
    ci = (avg - z*sem, avg + z*sem)
    return [avg, std, ci]

def noisy_data(data, rank, num_trials = 50, num_iter = 1000):
    """ Plot d(data, data + kR), where the entries in R are sampled i.i.d. from uniff([0,1]) averaged over num_trials trials 
    where k is in [0,1]"""
    # Keep track of average errors and confidence intervals
    errors = [[], []]
    high_cis = [[], []]
    low_cis = [[], []]

    # Every loop we add an addiitonal 0.1R
    for pct in range(10):
        pct /= 10
        # Keep track of the errors for each trial
        trial_errors = []
        chamfer_errors = []
        for trial in range(num_trials):
            R = np.random.uniform(size=X.shape)
            trial_errors.append(sim(data, data+pct*R, rank, num_iter = num_iter)/SIM_NORM_FACTOR)
            chamfer_errors.append(compute_chamfer_dist(data, data+pct*R)/CHAMFER_NORM_FACTOR)
            print(trial)

        # Compute the average and confidence interval
        error_avg, _, (low_ci, high_ci) = listStats(trial_errors)
        low_cis[0].append(low_ci)
        high_cis[0].append(high_ci)
        errors[0].append(error_avg)

        # Repeat for Chamfer
        error_avg, _, (low_ci, high_ci) = listStats(chamfer_errors)
        low_cis[1].append(low_ci)
        high_cis[1].append(high_ci)
        errors[1].append(error_avg)            
        errors.append(error_avg)
    return (errors[0], high_cis[0], low_cis[0], '#324dbe', 'jNMF Distance Measure'), (errors[1], low_cis[1], high_cis[1], "#be324d", 'Chamfers Distance Measure')



def large_subset(data, rank, num_trials = 50, num_iter=1000):
    '''Plot d(data, data*) where data* is data with some columns randomly removed averaged over num_trials trials'''
    # Set the smallest number of columns we try, the number of data points we plot, and how many columns we remove each time we move on to the next data point.
    smallest = 226
    num_steps = 6
    num_to_remove = 5

    # Keep track of average errors and confidence intervals
    errors = [[], []]
    high_cis = [[], []]
    low_cis = [[], []]

    # Compute each data point 
    for step in range(1, num_steps+1):
        # Keep track of errors for this number of columns removed in different trials
        trial_errors = []
        chamfer_errors = []        
        for trial in range(num_trials):
            # Remove columns from a copy of the data
            smaller_data = deepcopy(data)
            for n in range(num_to_remove * step):
                index = random.randint(0, smaller_data.shape[1]-1)
                smaller_data = np.delete(smaller_data, index, 1)
            # Add the distance to the errors for this size matrix
            trial_errors.append(sim(data, smaller_data, rank, num_iter=num_iter)/CHAMFER_NORM_FACTOR)
            chamfer_errors.append(compute_chamfer_dist(data, smaller_data)/CHAMFER_NORM_FACTOR)            
        # Compute the average and convidence intervals over the trials
        error_avg, _, (low_ci, high_ci) = listStats(trial_errors)
        low_cis[0].append(low_ci)
        high_cis[0].append(high_ci)
        errors[0].append(error_avg)
        # Repeat for Chamfer
        error_avg, _, (low_ci, high_ci) = listStats(chamfer_errors)
        low_cis[1].append(low_ci)
        high_cis[1].append(high_ci)
        errors[1].append(error_avg)

        print(trial)

    # Reverse the data so that we go from smaller matrices to bigger matrices, reading left to right
    for i in [0, 1]:
        errors[i].reverse()
        low_cis[i].reverse()
        high_cis[i].reverse()
    return (errors[0], low_cis[0], high_cis[0],'#324dbe', 'jNMF Distance Measure'), (errors[1], low_cis[1], high_cis[1], "#be324d", 'Chamfers Distance Measure')



def run_experiments(data, rank, num_trials=50):
    # For scaled experiment
    lambda_val = [0.1, 1, 10, 100]
    # For permutation experiment
    perm_data = deepcopy(data)
    rng = np.random.default_rng()


    # Keep track of average values
    # Store the average values of sim in the first entry, the average values of chamfer in the second
    self_sum = [0, 0]
    scaled_sum = [0, 0]
    permuted_sum = [0, 0]
    large_subset_sum = [0, 0]
    noise_sum = [0, 0]
    random_sum = [0, 0]
    for i in range(num_trials):
        # Self, identity, and permuted
        self_sum[0] += sim(data, data, rank)/SIM_NORM_FACTOR
        self_sum[1] += compute_chamfer_dist(data, data)/CHAMFER_NORM_FACTOR
        scaled_sum[0] += sim(data, np.random.choice(lambda_val) * data, rank)/SIM_NORM_FACTOR        
        scaled_sum[1] += compute_chamfer_dist(data, np.random.choice(lambda_val) * data)/CHAMFER_NORM_FACTOR
        rng.shuffle(data, axis = 1) # Permute the columns of the matrix
        permuted_sum[0] += sim(data, perm_data, rank)/SIM_NORM_FACTOR        
        permuted_sum[1] += compute_chamfer_dist(data, perm_data)/CHAMFER_NORM_FACTOR

        # Create a matrix with 26 columns randomly  removed
        smaller_data = deepcopy(data)
        for _ in range(26):
            index = random.randint(0, smaller_data.shape[1]-1)
            smaller_data = np.delete(smaller_data, index, 1)
        large_subset_sum[0] += sim(data, smaller_data, rank)/SIM_NORM_FACTOR            
        large_subset_sum[1] += compute_chamfer_dist(data, smaller_data)/CHAMFER_NORM_FACTOR

        # Update random and noisy experiments 
        R = np.random.uniform(size=X.shape)        
        noise_sum[0] += sim(data, data + R, rank)/SIM_NORM_FACTOR
        noise_sum[1] += compute_chamfer_dist(data, data + R)/CHAMFER_NORM_FACTOR
        random_sum[0] += sim(data, R, rank)/SIM_NORM_FACTOR        
        random_sum[1] += compute_chamfer_dist(data, R)/CHAMFER_NORM_FACTOR
    data = [self_sum, scaled_sum, permuted_sum, large_subset_sum, noise_sum, random_sum]
    return [datum[0]/num_trials for datum in data], [datum[1]/num_trials for datum in data]


#### Creat Plots of Data ####
def plot_data_with_noise(data: list) -> None:
    # Plot the data with confidence intervals
    plt.style.use('ggplot')
    fig = plt.figure()
    for errors, low_cis, high_cis, color, label in data:
        fig.gca().plot([x/10 for x in range(10)], errors, color=color, label=label)
        # fig.gca().fill_between([x/10 for x in range(10)], low_cis, high_cis, color='#324dbe', alpha=0.15)
        fig.gca().plot([x/10 for x in range(10)], low_cis, color=color, linestyle='dotted')
        fig.gca().plot([x/10 for x in range(10)], high_cis, color=color, linestyle='dotted')
    plt.legend()
    plt.xlabel('$\epsilon$')
    plt.ylabel('$d(X_1, X_1 + \epsilon N)$')
    plt.savefig('x_with_random.eps')
    plt.savefig('x_with_random.png')    
#     plt.show()


def plot_large_subset(data: list) -> None:
    # Plot the data
    smallest = 226
    num_steps = 6
    num_to_remove = 5

    plt.style.use('ggplot')
    fig = plt.figure()
    for errors, low_cis, high_cis, color, label in data:
        fig.gca().plot([(smallest + num_to_remove * i) * 100/256 for i in range(num_steps)], errors, color=color, label=label)  # This gives the percentage of columns we kept
        fig.gca().plot([(smallest + num_to_remove * i) * 100/256 for i in range(num_steps)], low_cis, color=color, linestyle='dotted')
        fig.gca().plot([(smallest + num_to_remove * i) * 100/256 for i in range(num_steps)], high_cis, color=color, linestyle='dotted')
    plt.legend()
    plt.xlabel('$q$')
    plt.ylabel('$d(X_1, X_2)$')
    plt.savefig('largesubset.eps')
    plt.savefig('largesubset.png')    
#     plt.show()


if __name__ == '__main__':
    data = loadmat('swimmer.mat')
    X = data['X']
    rank = 10
    CHAMFER_NORM_FACTOR = 1
    SIM_NORM_FACTOR = 1    
    # print(run_experiments(X, rank))
    # print(run_experiments_c(X, rank))
    # plot_large_subset(large_subset(X, rank, num_trials=50))
    plot_data_with_noise(noisy_data(X, rank, num_trials=50))
    # large_subset_c(X, rank)
    # plot_X_with_random_c(X, rank)
    # Y = 1 - X
    # error1, error2 = 0, 0
    # for _ in range(50):
    #     error1 += sim(X, Y, 10)/50
    #     error2 += compute_chamfer_dist(X, Y)/50        
    # print(error1, error2)
