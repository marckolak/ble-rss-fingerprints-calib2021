## Basic methods for scan preprocessing
## Author: Marcin Kolakowski

import numpy as np

def scan2xy(scan, x0=np.array([0, 0])):
    """ Convert a scan in angle-distance format to x,y  where 0,0 is the location of the scanner

    Parameters
    ----------
    scan: ndarray
        scan in angle-distance format [angle, dist]
    x0: ndarray, optional
        scanner coordinates, default [0,0]

    Returns
    -------
    xy_scan: ndarray
        scan converted  to xy coordinates
    """
    x = scan[:, 1] * np.cos(scan[:, 0])
    y = scan[:, 1] * np.sin(scan[:, 0])

    return np.c_[x, y] + x0


def xy2scan(xy):
    """ Convert a scan in x,y to angle-distance format

    Parameters
    ----------
    xy: ndarray
        set of points (x,y) constituting the scan

    Returns
    -------
    ar_scan: ndarray
        scan in angle-distance format [angle, dist]

    """
    a = np.arctan2(xy[:, 0], xy[:, 1])
    r = np.linalg.norm(xy, axis=1)

    scan = np.c_[a, r]

    return scan
