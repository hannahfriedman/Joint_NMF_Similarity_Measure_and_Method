from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from chamferdist import ChamferDistance


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
    avg = sum([np.linalg.norm(m[:,col]) for col in range(m.shape[1])])/m.shape[1]
    for col in range(m.shape[1]):
        if (np.linalg.norm(m[:, col])) >= 0.05 * avg:
            new[:, col] = m[:, col]/(np.linalg.norm(m[:, col]))
    return new


def compute_chamfer_dist(X: np.array, Y: np.array) -> float:
    ''' Compute the Chamfer distance between two arrays X and Y'''
    X = normalize(X)
    Y = normalize(Y)

    # We have to reshape the data into 3d tensors for the metric to work
    if len(X.shape) == 2:
        X = np.reshape(X, (*(X.shape), 1))
    if len(Y.shape) == 2:
        Y = np.reshape(Y, (*(Y.shape), 1))

    # Compute and return distance
    x_cloud = torch.tensor(X).float()
    y_cloud = torch.tensor(Y).float()
    chamferDist = ChamferDistance()
    dist_forward = chamferDist(x_cloud, y_cloud, bidirectional = True)
    return dist_forward.detach().cpu().item()
