from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import random

def sim(m1, m2, rank, verbose = False, num_iter = 1000, image_name = None, basis_shape = (11,20)):
    ''' Use jNMF to decompose matrices into A, S1, S2 and measure similarity between rows of S1 and S2
        Randomly sample values from 0 to max(max(S1[i]), max(S2[i]) and, for each value, gives the difference in numbers above that value
    '''
    # Perform Joint NMF
    m1 = normalize(m1)
    m2 = normalize(m2)
    combined = np.hstack((m1,m2))
    model = NMF(n_components=rank)
    A = model.fit_transform(combined)
    S = model.components_
    S1 = S[:, :m1.shape[1]]
    S2 = S[:, m1.shape[1]:]    

    # Visualize basis elements
    if type(image_name) == str:
        for i in range(A.shape[1]):
            plt.style.use('default')
            plt.matshow(np.reshape(A[:, i], basis_shape), cmap='gray')
            plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labeltop = False)
            plt.savefig(image_name + str(i))
            plt.show()

    # Compute similarity vector
    sim_vec = []
    # Compute each entry of the similarity vector
    for row in range(S1.shape[0]):
        # Find the range in which to sample
        max_val = max(np.amax(S1[row]), np.amax(S2[row]))
        total = 0
        for _ in range(num_iter):
            cutoff = random.uniform(0, max_val)
            S1_num_above = 0
            S2_num_above = 0
            # Count how many entries are above the cutoff
            for col in S1[row]:
                if col > cutoff:
                    S1_num_above += 1
            for col in S2[row]:
                if col > cutoff:
                    S2_num_above += 1
            # Add the difference in percentages to the average
            total += (S1_num_above/S1.shape[1]) - (S2_num_above/S2.shape[1])
        sim_vec.append(total/num_iter)
    if verbose:
        print(np.round(sim_vec, 3))
    return np.linalg.norm(sim_vec, ord=1)


def normalize(m):
    ''' Given a matrix m, return a new matrix which has all the columns of m scaled so that their mean is 1'''
    new = np.zeros(m.shape)
    for col in range(m.shape[1]):
        new[:, col] = m[:, col]/(np.linalg.norm(m[:, col]))
    return new
