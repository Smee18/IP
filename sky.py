import numpy as np
from scipy.optimize import curve_fit

def fit_gauss(img, bins=None, saturate_p=100, p0=None):
    """ Fits a Gaussian to the pixel distribution of an image.
    
    PARAMETERS
    ----------
    img : arraylike
        image to measure
    bins : None or arraylike or int, default None
        number of bins for estimating pixel distribution to fit. If None,
        chosen automatically based on size of image.
    saturate_p : float in (0,100], default 100
        percentile above which we ignore pixels to get a better fit 
    p0 : None or arraylike of length 3, default None
        preliminary guess for fit. If None, chosen automatically.
    
    RETURNS
    -------
    popt : tuple of floats
        best fit parameters to a Gaussian (a, mu, sigma)
    """
    flattened_img = img.flatten()
    
    if bins is None:
        bins = int(len(flattened_img)*saturate_p/100/200)
    if isinstance(bins, int) or isinstance(bins, np.integer):
        bins = np.linspace(np.nanmin(img), 
                           np.nanpercentile(img, saturate_p),
                           bins)
    
    hist, b = np.histogram(flattened_img, bins)
    bins = (bins[1:] + bins[:-1])/2
    
    if p0 is None:
        p0 = [np.nanmax(hist), bins[np.argmax(hist)], np.std(img)]
    popt, pcov = curve_fit(gauss, bins, hist, p0=p0)
    
    return popt

def gauss(x, a, mu, sigma):
    """ Gaussian function taking parameters as keyword args
    
    """
    return a*np.exp(-1*(x-mu)**2/(2*sigma**2))

def sigma_clip(data, alpha=3, tolerance=0.1, max_iterations=1000,
               verbose=False):
    """ Iterative sigma clipping function. Masks points outside alpha*sigma 
    of median and repeats until sigma of the clipped dataset is within 
    tolerance of the sigma of the previous iteration, or max_iterations has 
    been reached.
    
    Parameters
    ----------
    data : numpy.ndarray
        data to sigma clip
    alpha : float, default 3
        multiplier of sigma to clip at each iteration
    tolerance : float, default 0.1
        maximum allowed difference between sigma of current and previous 
        iteration for exit condition
    max_iterations : int, default 1000
        maximum number of iterations before exiting
    verbose : bool, default False
        whether to print information about the sigma-clipping process
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask to get sigma-clipped data
    """
    
    # initialise stuff
    mask = np.ones(data.shape, dtype=bool)
    diff = tolerance
    count = 0
    old_sig = 0
    
    # perform clipping loops
    while (diff >= tolerance) and (count < max_iterations):
        med = np.nanmedian(data[mask])
        sig = np.nanstd(data[mask])
        
        mask[np.abs(data - med) > sig*alpha] = False

        # update exit criteria
        diff = np.abs(old_sig - sig)/sig
        old_sig = sig
        count += 1
        
    if verbose:
        print('Input size:       %d' %(data.flatten().shape[0]))
        print('Number of clips:  %d' %count)
        print('Final data size:  %d' %(np.sum(mask)))
        print('Median:           %g' %(np.nanmedian(data[mask])))
        print('Mean:             %g' %(np.nanmean(data[mask])))
        print('Sigma:            %g' %(np.nanstd(data[mask])))
        
    return mask
