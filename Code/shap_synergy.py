import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import scipy.stats as st

rng = np.random.default_rng()


def shap_nsq_synergy(shap_values, shap_interaction_values):

    n = len(shap_values[0])
    nsq_syn_matrix=np.zeros((n, n))
    
    for j in range(n):
        for i in range(n):
            pi = shap_values[:, i]
            pij = shap_interaction_values[:, i, j]

            nsq_syn_matrix[j, i] = np.inner(pi, pij)/(np.linalg.norm(pi) * (np.linalg.norm(pij)))

    return nsq_syn_matrix


def bootsrap_nsq_syn_list(shap_values, shap_interaction_values, n_samples=10000, sample_size=100):

    nsq_list = []
    for _ in range(n_samples):
        
        shap_values_sample = rng.choice(shap_values, size=sample_size, replace=True)
        shap_interaction_values_sample = rng.choice(shap_interaction_values, size=sample_size, replace=True)
        
        nsq_list.append(shap_nsq_synergy(shap_values_sample, shap_interaction_values_sample))
        
    return np.asarray(nsq_list)


def pvalue0(distribution):

    return len(distribution[distribution<0])/len(distribution)



def bootstrap_synergy_values(X, explainer, n_samples=10000, sample_size=100, verbose=True, pvalue_method=pvalue0):

    synergies = np.mean(bootsrap_nsq_syn_list(X, explainer, n_samples=n_samples, sample_size=sample_size, verbose=verbose)**2, axis=0)
    pvalues = np.asarray([[pvalue_method(synergies[:, i, j]) for j in range(synergies.shape[1])] for i in range(synergies.shape[2])])

    return np.asarray([[(synergies[i, j], pvalues[i, j]) for j in range(synergies.shape[0])] for i in range(synergies.shape[0])])
