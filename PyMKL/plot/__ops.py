from typing import List, Union, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """Creates a color gradient between two colors.
    
    Args:
        c1 (str): Starting color (hex code or name)
        c2 (str): Ending color (hex code or name)
        mix (float): Mixing ratio between 0 (pure c1) and 1 (pure c2)
    
    Returns:
        str: Hex code of interpolated color
    """
    c1=np.array(mcolors.to_rgb(c1))
    c2=np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1-mix)*c1 + mix*c2)


def descriptors(descriptors: List[np.ndarray], groupby: int, figsize: Tuple[int,int] = (16,16), varnames: Union[np.ndarray, list] = None, dimnames: Union[np.ndarray, list] = None, same_scale: bool = True, return_axes: bool = False, grid: bool = True, color_from: Union[str, List[str]] = "blue", color_to: Union[str, List[str]] = "red", groupcolors: int = None, **kwargs):
    """Creates a multi-panel plot of descriptor values across features and dimensions.
    
    Args:
        descriptors: List of descriptor arrays to plot
        groupby: Number of groups to combine in each subplot
        figsize: Figure dimensions (width, height)
        varnames: Names for each feature/variable
        dimnames: Names for each dimension
        same_scale: Whether to use same y-axis scale within each feature
        return_axes: Whether to return figure and axes objects
        grid: Whether to show grid lines
        color_from: Starting color(s) for gradient
        color_to: Ending color(s) for gradient
        groupcolors: Number of distinct colors to use (defaults to groupby)
        **kwargs: Additional arguments passed to plt.subplots()
    
    Returns:
        Optional[Tuple[plt.Figure, np.ndarray]]: Figure and axes if return_axes is True
    """

    # Rest
    n_features = len(descriptors)
    n_dimensions = descriptors[0].shape[1]//groupby
    if groupcolors is None:
        groupcolors = groupby

    # Color range
    if isinstance(color_from, str):
        color_from = [color_from]*n_features
    if isinstance(color_to, str):
        color_to = [color_to]*n_features
    assert (len(color_from) == n_features) and (len(color_from) == n_features), "The number of specified colors must match the number of descriptors"
    color_range = [
        [colorFader(mcolors.to_hex(c_from),mcolors.to_hex(c_to),v) for v in np.linspace(0,1,groupcolors)] 
        for (c_from,c_to) in zip(color_from,color_to)
    ]

    # Initialize figure
    fig,ax = plt.subplots(nrows=n_features,ncols=n_dimensions,figsize=figsize,**kwargs)
    if ax.ndim == 1:
        ax = ax[:,None]
    for i,des in enumerate(descriptors):
        for j,val in enumerate(des.T):
            ax[i,j//groupby].plot(val,color=color_range[i][j%groupcolors])
            ax[i,j//groupby].set_xlim([0,val.size-1])
        
    # Set figure options
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]-1) for j in range(ax.shape[1])]
    [ax[i,j].set_yticks([]) for i in range(ax.shape[0])   for j in range(1,ax.shape[1])]
    if varnames is not None:
        [ax[i,0].set_ylabel(f"{varnames[i]}") for i in range(ax.shape[0])]
    else:
        [ax[i,0].set_ylabel(f"Feat. {i+1}") for i in range(ax.shape[0])]
    if dimnames is not None:
        [ax[0,j].set_title(f"{dimnames[j]}") for j in range(ax.shape[1])]
    else:
        [ax[0,j].set_title(f"Dim. {j+1}") for j in range(ax.shape[1])]
    # Set figure ylims
    if same_scale and "sharey" not in kwargs:
        for i in range(ax.shape[0]):
            ylims = [0,0]
            for j in range(ax.shape[1]):
                ylim = ax[i,j].get_ylim()
                ylims[0] = min([ylims[0],ylim[0]])
                ylims[1] = max([ylims[1],ylim[1]])
            
            for j in range(ax.shape[1]):
                ax[i,j].set_ylim(ylims)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01)
    fig.align_ylabels(ax[:,0])

    if return_axes:
        return fig,ax


def path(descriptors: List[np.ndarray], figsize=(16,16), varnames: Union[np.ndarray, list] = None, dimnames: Union[np.ndarray, list] = None):
    """Creates a multi-panel plot showing paths of descriptor values.
    
    Similar to the descriptors() function but specifically for visualizing paths
    or trajectories in the descriptor space, using a different color scheme.
    
    Args:
        descriptors: List of descriptor arrays to plot
        figsize: Figure dimensions (width, height)
        varnames: Names for each feature/variable
        dimnames: Names for each point/dimension
    """

    # Rest
    n_features = len(descriptors)
    n_dimensions = descriptors[0].shape[1]

    # Initialize figure
    fig,ax = plt.subplots(nrows=n_features,ncols=n_dimensions,figsize=figsize)
    for i,des in enumerate(descriptors):
        for j,val in enumerate(des.T):
            col = 1/5*(j%5)
            ax[i,j].plot(val,color=[col,0,1-col])
        
    # Set figure options
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]-1) for j in range(ax.shape[1])]
    if varnames is not None:
        [ax[i,0].set_ylabel(f"{varnames[i]}") for i in range(ax.shape[0])]
    else:
        [ax[i,0].set_ylabel(f"Feat. {i+1}") for i in range(ax.shape[0])]
    if dimnames is not None:
        [ax[0,j].set_title(f"Point {dimnames[j]}") for j in range(ax.shape[1])]
    else:
        [ax[0,j].set_title(f"Point {j+1}") for j in range(ax.shape[1])]
    [ax[i,j].set_xticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    [ax[i,j].set_yticks([]) for i in range(ax.shape[0]) for j in range(ax.shape[1])]
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01,wspace=0.01)
    fig.align_ylabels(ax[:,0])



