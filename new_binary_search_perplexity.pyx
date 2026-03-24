# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

# BSD 3-Clause License
# 
# Copyright (c) 2007-2024 The scikit-learn developers.
# All rights reserved.
#
# This file is part of scikit-learn and has been modified by Erez Peterfreund on 2025-07-21.
# Modifications:
# - Allowed the input distances to be either float32_t or float64_t.
# - Allowed looking at a consecutive subset of samples and their corresponding distances with all the samples.
# - Allowed to input betas as initialization points.
# - Forced the function to return the extracted betas.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np

cimport numpy as cnp 

from libc cimport math
from libc.math cimport INFINITY



ctypedef cnp.float32_t float32_t
ctypedef cnp.float64_t float64_t

#############################################################################################################

cdef float EPSILON_DBL = 1e-8
cdef float PERPLEXITY_TOLERANCE = 1e-5

ctypedef fused my_float:
    float32_t
    float64_t

def _new_binary_search_perplexity(
        my_float[:, :] sqdistances,
        float desired_perplexity,
        long ind_first_index=-1):
    """Binary search for sigmas of conditional Gaussians.


    Parameters
    ----------
    sqdistances : ndarray of shape (n_subset, n_samples), dtype=np.float32 or np.float64
        Pairwise distances between a group of samples and the full set of samples.  
        The group consists of samples that occur one after another in the dataset,  
        starting at index `ind_first_index`.

    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.
            
    ind_first_index : int
        The index in the full dataset where this group of consecutive samples begins.
    
    Returns
    -------
    P : ndarray of shape (n_subset, n_samples), dtype=np.float64
        Probabilities of conditional Gaussian distributions p_i|j.
    """
    # Maximum number of binary search steps
    cdef long n_steps = 100

    cdef long n_subset = sqdistances.shape[0]
    cdef long n_samples = sqdistances.shape[1]

    # Precisions of conditional Gaussian distributions
    cdef double beta
    cdef double beta_min
    cdef double beta_max

    # Use log scale
    cdef double desired_entropy = math.log(desired_perplexity)
    cdef double entropy_diff

    cdef double entropy
    cdef double sum_Pi
    cdef double sum_disti_Pi
    cdef long i, j, l
        
    cdef float64_t[:, :] P = np.zeros(
        (n_subset, n_samples), dtype=np.float64)

    cdef float64_t[:] new_betas = np.ones((n_subset,), dtype=np.float64)
        
    for i in range(n_subset):
        beta_min = -INFINITY
        beta_max = INFINITY
        
        beta= new_betas[i]

        # Binary search of precision for i-th conditional distribution
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed over all data
            sum_Pi = 0.0
            for j in range(n_samples):
                if (j != i and ind_first_index==-1)  or (ind_first_index+i!= j  and ind_first_index!=-1):
                    P[i, j] = math.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL
            sum_disti_Pi = 0.0

            for j in range(n_samples):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = math.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == INFINITY:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -INFINITY:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        new_betas[i] = beta

    return np.asarray(P), np.asarray(new_betas)
