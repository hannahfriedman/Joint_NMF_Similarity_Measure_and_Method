import matplotlib.pyplot as plt
import statistics as stats
import math as m
import random
import numpy as np
from copy  import deepcopy
from similarity import sim
from scipy.io import loadmat
plt.rc('text', usetex=True)

def listStats(numList):
    """takes in a list of numbers and returns [average, standard deviation, 95% confidence interval]"""
    avg = stats.mean(numList)
    std = stats.stdev(numList)
    sem = std / m.sqrt(len(numList))
    z = 1.96 # 95% ci
    ci = (avg - z*sem, avg + z*sem)
    return [avg, std, ci]

def plot_X_with_random(data, rank, num_trials = 50, num_iter = 1000):
    """ Plot d(data, data + kR), where the entries in R are sampled i.i.d. from uniff([0,1]) averaged over num_trials trials 
    where k is in [0,1]"""
    # Keep track of average errors and confidence intervals
    errors = []
    high_cis = []
    low_cis = []

    # Every loop we add an addiitonal 0.1R
    for pct in range(10):
        pct /= 10
        # Keep track of the errors for each trial
        trial_errors = []
        for trial in range(num_trials):
            R = np.random.uniform(size=X.shape)
            error = sim(data, data+pct*R, rank, num_iter = num_iter)
            trial_errors.append(error)
        # Compute the average and confidence interval
        error_avg, _, (low_ci, high_ci) = listStats(trial_errors)
        low_cis.append(low_ci)
        high_cis.append(high_ci)
        errors.append(error_avg)
    # Plot the data with confidence intervals
    print(low_cis, high_cis)
    plt.style.use('ggplot')
    fig = plt.figure()
    fig.gca().plot([x/10 for x in range(10)], errors, color='#324dbe')
    # fig.gca().fill_between([x/10 for x in range(10)], low_cis, high_cis, color='#324dbe', alpha=0.15)
    fig.gca().plot([x/10 for x in range(10)], low_cis, color='#324dbe', linestyle='dotted')
    fig.gca().plot([x/10 for x in range(10)], high_cis, color='#324dbe', linestyle='dotted')    
    plt.xlabel('$\epsilon$')
    plt.ylabel('$d(X_1, X_1 + \epsilon N)$')
    plt.savefig('x_with_random.eps')
    plt.show()


def large_subset(data, rank, num_trials = 50, num_iter = 1000):
    '''Plot d(data, data*) where data* is data with some columns randomly removed averaged over num_trials trials'''
    # Set the smallest number of columns we try, the number of data points we plot, and how many columns we remove each time we move on to the next data point.
    smallest = 226
    num_steps = 6
    num_to_remove = 5

    # Keep track of average errors and confidence intervals
    errors = []
    high_cis = []
    low_cis = []

    # Compute each data point 
    for step in range(1, num_steps+1):
        # Keep track of errors for this number of columns removed in different trials
        trial_errors = []
        for trial in range(num_trials):
            # Remove columns from a copy of the data
            smaller_data = deepcopy(data)
            for n in range(num_to_remove * step):
                index = random.randint(0, smaller_data.shape[1]-1)
                smaller_data = np.delete(smaller_data, index, 1)
            # Add the distance to the errors for this size matrix
            trial_errors.append(sim(data, smaller_data, rank, num_iter=num_iter))
        # Compute the average and convidence intervals over the trials
        error_avg, _, (low_ci, high_ci) = listStats(trial_errors)
        low_cis.append(low_ci)
        high_cis.append(high_ci)
        errors.append(error_avg)
    # Reverse the data so that we go from smaller matrices to bigger matrices, reading left to right
    errors.reverse()
    low_cis.reverse()
    high_cis.reverse()
    # Plot the data
    plt.style.use('ggplot')
    fig = plt.figure()
    fig.gca().plot([(smallest + num_to_remove * i) * 100/256 for i in range(num_steps)], errors, color='#324dbe')  # This gives the percentage of columns we kept
    # fig.gca().fill_between([(smallest + num_to_remove * i) * 100/256 for i in range(num_steps)], low_cis, high_cis, color='#324dbe', alpha=.15)
    fig.gca().plot([(smallest + num_to_remove * i) * 100/256 for i in range(num_steps)], low_cis, color='#324dbe', linestyle='dotted')
    fig.gca().plot([(smallest + num_to_remove * i) * 100/256 for i in range(num_steps)], high_cis, color='#324dbe', linestyle='dotted')    
    plt.xlabel('$q$')
    plt.ylabel('$d(X_1, X_2)$')
    plt.savefig('largesubset.eps')
    plt.show()


def run_experiments(data, rank):
    # Set how many trials we're averaging over
    num_trials = 50
    # For scaled experiment
    lambda_val = [0.1, 1, 10, 100]
    # For permutation experiment
    perm_data = deepcopy(data)
    rng = np.random.default_rng()
    # For random and noisy
    R = np.random.uniform(size=X.shape)
    # For noisy
    noisy = data + R

    # Keep track of average values
    self_sum = 0
    scaled_sum = 0
    permuted_sum = 0
    large_subset_sum = 0
    noise_sum = 0
    random_sum = 0
    for i in range(num_trials):
        self_sum += sim(data, data, rank)
        scaled_sum += sim(data, np.random.choice(lambda_val) * data, rank)
        rng.shuffle(data, axis = 1) # Permute the columns of the matrix
        permuted_sum += sim(data, perm_data, rank)
        # Create a matrix with 26 columns randomly  removed
        smaller_data = deepcopy(data)
        for _ in range(26):
            index = random.randint(0, smaller_data.shape[1]-1)
            smaller_data = np.delete(smaller_data, index, 1)
        large_subset_sum += sim(data, smaller_data, rank)
        noise_sum += sim(data, data + R, rank)
        random_sum += sim(data, R, rank)
    return self_sum/num_trials, scaled_sum/num_trials, permuted_sum/num_trials, large_subset_sum/num_trials, noise_sum/num_trials, random_sum/num_trials

if __name__ == '__main__':
    data = loadmat('swimmer.mat')
    X = data['X']
    rank = 10
    # print(run_experiments(X, rank))
    # large_subset(X, rank)
    plot_X_with_random(X, rank)
    
