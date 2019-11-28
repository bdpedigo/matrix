import numpy as np


def unique_by_size(data):
    """Equivalent to np.unique but returns data in order sorted by frequency of values
    
    Parameters
    ----------
    data : np.ndarray
        array on which to find unique values
    
    Returns
    -------
    np.ndarray
        unique elements in `data` sorted by frequency, with the most observations first
    
    np.ndarray
        counts of the unique elements in `data`
    """
    unique_data, counts = np.unique(data, return_counts=True)
    sort_inds = np.argsort(counts)[::-1]  # reverse order to get largest class first
    unique_data = unique_data[sort_inds]
    counts = counts[sort_inds]
    return unique_data, counts

