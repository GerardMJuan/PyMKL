from typing import List, Callable, Dict, Tuple, Union, Any

import math
import numpy as np
import scipy as sp
import numpy.matlib
import sak

from scipy.spatial.distance import pdist, cdist, squareform
from pygam import LinearGAM, s

KERNEL_LIST = (
    "euclidean",
    "euclidean_density",
    "categorical",
    "ordinal",
    "xcorr",
    "euclidean_xcorr",
    "default",
    "rbf",
    "polynomial",
    "euclidean_age",
)


def euclidean_age(x: np.ndarray, 
                y: np.ndarray = None, 
                knn: int = None, 
                alpha: float = -1.0,
                gestational_ages: np.ndarray = None,
                gestational_ages_y: np.ndarray = None,
                sigma_age: float = None,
                **kwargs) -> Tuple[np.ndarray, float, float]:
    """
    Creates an age-normalized kernel matrix for a single feature.
    
    The kernel computation incorporates age-specific normalization to compare subjects
    based on their deviation from age-specific norms rather than absolute values.
    
    Args:
        x: Input data array of shape (n_samples, n_features)
        y: Optional second input array for asymmetric kernel (defaults to x)
        gestational_ages: Gestational ages for x samples
        gestational_ages_y: Gestational ages for y samples (if y is provided)
        knn: Number of nearest neighbors for kernel bandwidth
        alpha: Kernel parameter controlling the decay
        sigma_age: Bandwidth for age weighting (if None, computed from data)
        **kwargs: Additional parameters passed to kernel functions
    
    Returns:
        Tuple containing:
        - Kernel matrix (n_samples_x, n_samples_y)
        - Kernel variance
        - Characteristic length scale (sigma)
    """
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Input must be 1D or 2D array")

    # If y is None, copy x
    if y is None:
        use_pdist = True
        y = x.copy()
        gestational_ages_y = gestational_ages
    else:
        y = y.copy().squeeze()
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("Input must be 1D or 2D array")
        use_pdist = False
        if gestational_ages_y is None:
            raise ValueError("gestational_ages_y must be provided when y is provided")

    # Get dimensions
    N = x.shape[0]

    # Apply default number of nearest neighbors
    if knn is None:
        knn = math.floor(np.sqrt(N))

    if gestational_ages is not None:
        # Fit GAM model for x data
        n_features = x.shape[1]
        
        normalized_x = np.zeros_like(x)
        
        # Fit separate GAM model for each feature
        for i in range(n_features):
            gam_x = LinearGAM(s(0))  # Single smooth term for age
            gam_x.gridsearch(gestational_ages.reshape(-1, 1), x[:, i], progress=False)
            age_means_x = gam_x.predict(gestational_ages.reshape(-1, 1))
            normalized_x[:, i] = x[:, i] - age_means_x

        # If y is different, normalize it separately
        if not use_pdist:
            gam_y = LinearGAM()
            gam_y.gridsearch(gestational_ages_y.reshape(-1, 1), y)
            age_means_y = gam_y.predict(gestational_ages_y.reshape(-1, 1))
            normalized_y = y - age_means_y[:, None]
        else:
            normalized_y = normalized_x

        # Compute age weights
        if sigma_age is None:
            # Compute sigma_age as standard deviation of age differences
            sigma_age = np.std(gestational_ages)
            
        # Compute age differences matrix
        age_diffs = np.abs(gestational_ages[:, None] - gestational_ages_y[None, :])
        
        # Compute age weights using a Gaussian function
        age_weights = np.exp(- (age_diffs ** 2) / (2 * sigma_age ** 2))
    else:
        normalized_x = x
        normalized_y = y
        age_weights = 1.0

    # Obtain pairwise distances
    if use_pdist:
        distances = squareform(pdist(normalized_x, metric="euclidean"))
    else:
        distances = cdist(normalized_x, normalized_y, metric="euclidean")

    # Modify distances by age weights if provided
    if gestational_ages is not None:
        weighted_distances = distances * age_weights
    else:
        weighted_distances = distances

    # Obtain infinite diagonal distances
    inf_distances = weighted_distances.copy()
    inf_distances[inf_distances < np.finfo(weighted_distances.dtype).eps] = np.inf

    # Sort distances and compute sigma based on kNN
    inf_distances_sorted = np.sort(inf_distances, axis=0)
    sigma = np.mean(inf_distances_sorted[: min([knn, N]), :])

    # Compute kernel
    K = np.exp(alpha * (np.square(weighted_distances) / (2. * (sigma) ** 2.)))
    var = np.var(K)

    return K, var, sigma


def rbf(
    x: np.ndarray,
    y: np.ndarray = None,
    knn: int = None,
    alpha: float = -1.0,
    **kwargs
):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("RBF kernel must take 1D or 2D inputs")

    # If y is None, copy x
    if y is None:
        use_pdist = True
        y = x.copy()
    else:
        y = y.copy().squeeze()
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("RBF kernel must take 1D or 2D inputs")
        use_pdist = False

    # Get dimensions
    N = x.shape[0]

    # Apply default number of nearest neighbors
    if knn is None:
        knn = math.floor(np.sqrt(N))

    # Obtain pairwise distances
    if use_pdist:
        distances = squareform(pdist(x, metric="euclidean"))
    else:
        distances = cdist(x, y, metric="euclidean")

    # Obtain infinite diagonal distances
    inf_distances = distances.copy()
    inf_distances[inf_distances < np.finfo(distances.dtype).eps] = np.inf

    # Sort these distances and retrieve the <knn>-th most similar elements for computing sigma
    inf_distances_sorted = np.sort(inf_distances, axis=0)
    sigma = np.mean(inf_distances_sorted[: min([knn, N]), :])

    # Compute gamma
    gamma = -alpha / (2 * sigma ** 2)

    # Compute the RBF kernel
    K = np.exp(-gamma * np.square(distances))
    var = np.var(K)

    return K, var, sigma


def polynomial(
    x: np.ndarray,
    y: np.ndarray = None,
    degree: int = 3,
    alpha: float = 1.0,
    coef0: float = 1.0,
    **kwargs
):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Polynomial kernel must take 1D or 2D inputs")

    # If y is None, copy x
    y = x.copy() if y is None else y.copy().squeeze()
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError("Polynomial kernel must take 1D or 2D inputs")

    # Compute the dot product
    K = np.dot(x, y.T)

    # Compute the polynomial kernel
    K = (alpha * K + coef0) ** degree
    var = np.var(K)
    sigma = 1.0  # Not used for polynomial kernel

    return K, var, sigma


def euclidean_xcorr(x: np.ndarray, y: np.ndarray = None, knn: int = None, alpha: float = -1, maxlags: int = 0, **kwargs):
    # Obtain pairwise distances
    K_eucl,_,_ = euclidean(x,y,knn,alpha,**kwargs)
    K_corr,_,_ = xcorr(x,y,**kwargs)
    ptg = kwargs.get("proportion",0.5)
    K = ptg*K_eucl + (1-ptg)*K_corr
    var = np.var(K)

    return K,var,1


def xcorr(x: np.ndarray, y: np.ndarray = None, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("xcorr kernel must take 1D or 2D inputs")

    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()
        if y.ndim == 1:
            y = y[:,None]
        if y.ndim != 2:
            raise ValueError("euclidean_xcorr kernel must take 1D or 2D inputs")

    # Obtain pairwise distances
    K = (1+sak.signal.xcorr(x,y,maxlags=0)[0].squeeze())/2
    var = np.var(K)

    return K,var,1


def euclidean(
    x: np.ndarray,
    y: np.ndarray = None,
    knn: int = None,
    alpha: float = -1,
    **kwargs
):
    x = x.copy() # .squeeze()
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Euclidean kernel must take 1D or 2D inputs")

    # If y is None, copy x
    if y is None:
        use_pdist = True
        y = x.copy()
        gestational_ages_y = kwargs.get('gestational_ages', None)
    else:
        y = y.copy() # .squeeze()
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("Euclidean kernel must take 1D or 2D inputs")
        use_pdist = False
        gestational_ages_y = kwargs.get('gestational_ages_y', None)

    # Get gestational ages and sigma_age from kwargs
    gestational_ages = kwargs.get('gestational_ages', None)
    sigma_age = kwargs.get('sigma_age', None)
    # if gestational_ages is None:
    #     raise ValueError("gestational_ages must be provided for soft age weighting")

    # Get dimensions
    N = x.shape[0]

    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))

    # Obtain pairwise distances
    if use_pdist:
        distances = squareform(pdist(x, metric="euclidean"))
        if gestational_ages is not None:
            # Compute age differences
            age_diffs = np.abs(gestational_ages[:, None] - gestational_ages[None, :])
    else:
        distances = cdist(x, y, metric="euclidean")
        if gestational_ages is not None:
            # Compute age differences
            age_diffs = np.abs(gestational_ages[:, None] - gestational_ages_y[None, :])

    if gestational_ages is not None:

        # Compute age weights
        if sigma_age is None:
            # Compute sigma_age as standard deviation of age differences
            sigma_age = np.std(gestational_ages)
        age_weights = np.exp(- (age_diffs ** 2) / (2 * sigma_age ** 2))

        # Modify distances by age weights
        weighted_distances = distances * age_weights
    else:
        weighted_distances = distances

    # Obtain inf-diagonal distances
    inf_distances = weighted_distances.copy()
    inf_distances[inf_distances < np.finfo(weighted_distances.dtype).eps] = np.inf

    # Sort these distances and retrieve the <knn>-th most similar elements for computing sigma
    inf_distances_sorted = np.sort(inf_distances, axis=0)
    sigma = np.mean(inf_distances_sorted[: min([knn, N]), :])

    # Obtain kernel value
    K = np.exp(alpha * (np.square(weighted_distances) / (2. * (sigma) ** 2.)))
    var = np.var(K)

    return K, var, sigma

def compute_age_weights(gestational_ages: np.ndarray, sigma_age: float):
    N = len(gestational_ages)
    # Compute pairwise gestational age differences
    age_diffs = np.abs(gestational_ages[:, None] - gestational_ages[None, :])
    # Compute age weights using a Gaussian function
    age_weights = np.exp(- (age_diffs ** 2) / (2 * sigma_age ** 2))
    return age_weights


def euclidean_density(x: np.ndarray, y: np.ndarray = None, knn: int = None, alpha: float = -1, **kwargs):
    x = x.copy().squeeze()
    if x.ndim == 1:
        x = x[:,None]
    if x.ndim != 2:
        raise ValueError("Euclidean kernel must take 1D or 2D inputs")

    # If y is None, copy x
    if y is None:
        use_pdist = True
        y = x.copy()
    else:
        y = y.copy().squeeze()
        if y.ndim == 1:
            y = y[:,None]
        if y.ndim != 2:
            raise ValueError("euclidean_xcorr kernel must take 1D or 2D inputs")
        use_pdist = False

    # Get dimensions
    N = x.shape[0]

    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))

    # Obtain pairwise distances
    if use_pdist:
        distances = squareform(pdist(x,metric="euclidean"))
    else:
        distances = cdist(x,y,metric="euclidean")

    # Obtain inf-diagonal distances
    inf_distances = distances.copy()
    inf_distances[inf_distances < np.finfo(distances.dtype).eps] = np.inf

    # Sort these distances and retrieve the <knn>-th most similar elements for computing sigma
    inf_distances_sorted = np.sort(inf_distances,axis=0)
    sigma = np.mean(inf_distances_sorted[:min([knn,N]),:],0)
    sigma[sigma == 0] = np.min(sigma[sigma > 0])
    sigma = np.matlib.repmat(sigma,N,1)

    # Obtain kernel value
    K = np.exp(alpha*(np.square(distances) / (2.*(sigma)**2.)))
    var = np.var(K)

    return K,var,sigma


def categorical(x: np.ndarray, y: np.ndarray = None, alpha: float = 1.0, random: float = 0.0, zero_method: str = "min", eye: bool = False, *args, **kwargs):
    """https://upcommons.upc.edu/bitstream/handle/2099.1/17172/MarcoVillegas.pdf"""
    x = x.copy().squeeze()
    if x.ndim > 1:
        raise ValueError("Categorical kernel must take 1D inputs")
    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()

    x = x.astype(int)
    y = y.astype(int)

    # Count occurrences of each category    
    counts_x = np.bincount(x)
    unique_x = np.unique(x)
    counts_y = np.bincount(y)
    unique_y = np.unique(y)

    # Take out zero if present (why, numpy, why)
    if counts_x.size > unique_x.size:
        counts_x = counts_x[1:]
    if counts_y.size > unique_y.size:
        counts_y = counts_y[1:]

    # Compute probability of each category in population
    prob_x,prob_y = np.zeros((len(x),)),np.zeros((len(y),))
    for i,u in enumerate(unique_x):
        prob_x[x == u] = counts_x[i]/len(x)
    for i,u in enumerate(unique_y):
        prob_y[y == u] = counts_y[i]/len(y)
    prob = np.sqrt((prob_x[:,None] * prob_y[None,:]))

    # Compute kernel
    K = (x[:,None] == y[None,]) * (1 - prob**alpha)**(1/alpha)

    # Add values when proba is zero (modified from Marco Villegas)
    if zero_method == "min":
        K[K == 0] = np.clip(np.min(prob)/2,0,1)
    else:
        K[K == 0] = np.clip(np.min(prob)-0.05,0,1)
    
    # [EXPERIMENTAL] Give some leeway to slightly random values
    if random != 0:
        K = np.clip(K*np.random.uniform(1-random,1+random,K.shape), a_min=0, a_max=1)

    # [EXPERIMENTAL] Fill diagonal (distance to self is zero)
    if eye:
        np.fill_diagonal(K,np.ones((K.shape[0],1)))

    return K, 1, 1


def ordinal(x: np.ndarray, y: np.ndarray = None, *args, **kwargs):
    x = x.copy().squeeze()
    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()
    if x.ndim != y.ndim:
        raise ValueError("x and y vectors have different depth")

    # Compute kernel
    if x.ndim == 1:
        x_range = np.max(np.concatenate((x,y))) - np.min(np.concatenate((x,y)))
        distances = np.abs(x[:,None] - y[None,])
    elif x.ndim == 2:
        distances = cdist(x,y,metric="euclidean")
        x_range = np.max(distances) - np.min(distances)
    else:
        raise ValueError("Ordinal kernel must take 1D or 2D inputs")

    K = (x_range - distances)/x_range

    return K,1.0,1.0


def default(x: np.ndarray, y: np.ndarray = None, *args, **kwargs):
    x = x.copy().squeeze()
    if x.ndim != 1:
        raise ValueError("Categorical kernel must take 1D inputs")

    # If y is None, copy x
    if y is None:
        y = x.copy()
    else:
        y = y.copy().squeeze()

    if x.ndim != y.ndim:
        raise ValueError("x and y vectors have different depth")

    # Compute kernel
    K = (x[:,None] == y[None,]).clip(min=0.9)

    return K,1,1


def kernel_stack(
    X: Union[List[np.ndarray], Dict[Any, np.ndarray]],
    kernel: Union[str, List[str]] = "euclidean",
    knn: int = None,
    alpha: float = -1,
    return_sigmas: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Obtain kernels from list of features through a metric. Current metric is 
    squareform(pdist(x,metric="euclidean")), but any metric that operates on a 
    matrix M ∈ [S x L], where S is the number of samples in the population and L
    is the length of the sample, is valid. Moreover, custom metrics can operate
    with features that contain samples with different lengths in a List[List[float]]
    fashion.

    Inputs:
    * X:        <List[np.ndarray]> (or <List[List[float]]>) if custom metric is provided
    * metric:   <Callable>, so that it takes as input a <np.ndarray> or a <List[float]>, 
                according to what was provided in X.
    * knn:      <int>, number of nearest neighbours for the computation of the sigma

    For further details, refer to https://doi.org/10.1016/j.media.2016.06.007 and to 
    https://doi.org/10.1109/TPAMI.2010.183
    """

    # Retrieve dimensions
    M = len(X) # Number of different features to work with
    if isinstance(X,List):
        N = X[0].shape[0] # Number of samples in the population
    elif isinstance(X,Dict):
        keys = list(X)
        N = X[keys[0]].shape[0]
    
    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))

    # Check kernel in valid kernels
    per_kernel = False
    if isinstance(kernel, str):
        if kernel not in KERNEL_LIST:
            raise ValueError(f"Valid kernels are: {KERNEL_LIST}")
        kernel = eval(kernel)
        per_kernel = False
    elif isinstance(kernel, (list, tuple)):
        for k in kernel:
            if k not in KERNEL_LIST:
                raise ValueError(f"Valid kernels are: {KERNEL_LIST}. Provided kernel: {k}")
        kernel = [eval(k) for k in kernel]
        assert len(kernel) == len(X), "Invalid kernel configuration. Must be of the same size of X"
        per_kernel = True
    else:
        raise ValueError(f"Kernel type not supported. Valid kernels are: {KERNEL_LIST} or a list of them")
        
    # Create matrix
    K = np.zeros((M, N, N), dtype='float64')
    var = np.zeros((M,), dtype='float64')
    sigmas = np.zeros((M, N, N), dtype='float64')

    # Compute pairwise distances
    for m, k in enumerate(X):
        # Retrieve the feature according to its input data type
        if isinstance(X, List):
            feature = X[m]
        elif isinstance(X, Dict):
            feature = X[k]
        # Obtain kernel value
        if per_kernel:
            K[m], var[m], sigmas[m] = kernel[m](feature, knn=knn, alpha=alpha, **kwargs)
        else:
            K[m], var[m], sigmas[m] = kernel(feature, knn=knn, alpha=alpha, **kwargs)
    if return_sigmas:
        return K, var, sigmas
    else:
        return K, var


def get_W_and_D(K: np.ndarray, var: np.ndarray, knn: int = None):
    # Get number of features and samples
    M = K.shape[0]
    N = K.shape[1]
    
    # Apply default number of nearest neighbours
    if knn is None:
        knn = math.floor(np.sqrt(N))
        
    # Get minimum variance
    min_var = np.min(var)
    
    # Copy the kernels tensor to avoid modifying it
    aux = np.copy(K)
    
    # Normalize features by variance
    for m in range(M):
        alfa_m = var[m]*1.0/min_var
        aux[m] = np.power(aux[m], 1/alfa_m)
        
    # Global affinity matrix: mean of the normalized affinity matrices
    W = np.sum(aux, axis=0) / M  
    sparse_W = np.zeros_like(W, dtype='float64')
    for i,r in enumerate(W):
        s = r.argsort()[::-1]
        ind_min = s[knn + 1:]
        r[ind_min] = 0
        sparse_W[i] = r
    sparse_W = np.maximum(sparse_W, sparse_W.transpose())

    # Diagonal matrix - in each diagonal element is the sum of the corresponding row
    D = np.sum(sparse_W, axis=0)[:,None]

    return sparse_W, D



def compute_test_kernels(test_features: List[np.ndarray], 
                        train_features: List[np.ndarray],
                        kernel_type: str = 'euclidean',
                        knn: int = None,
                        alpha: float = -1,
                        **kernel_params) -> np.ndarray:
    """
    Computes kernel matrices between test samples and training samples.
    
    Args:
        test_features: List of M feature arrays, each (n_test x d)
        train_features: List of M feature arrays, each (n_train x d) 
        kernel_type: Type of kernel to use
        knn: Number of nearest neighbors for kernel bandwidth
        alpha: Kernel parameter controlling the decay
        kernel_params: Additional kernel parameters
    
    Returns:
        K_test: Array of shape (M, n_test, n_train) containing kernel values
    """
    M = len(test_features)  # Number of features
    n_test = test_features[0].shape[0]
    n_train = train_features[0].shape[0]
    
    # Initialize kernel matrices
    K_test = np.zeros((M, n_test, n_train))
    
    # For each feature
    for m in range(M):
        # For each test sample
        for i in range(n_test):
            # Compute kernel between test sample i and all training samples
            # test_features[m][i] is a single sample (1 x d)
            # train_features[m] contains all training samples (n_train x d)
            
            if kernel_type == 'euclidean':
                # Compute euclidean distances 
                K, var, sigma = euclidean(test_features[m][i:i+1], train_features[m], knn, alpha)

            # add to K_test
            K_test[m, i, :] = K.squeeze()

    return K_test