"""
The wild_thumper module contains functions loading and processing output files saved by
the wild_thumper controller. The functions allow the user to decode the robot motion and LiDAR
gathered with the Scanse Sweep sensor.

See more at https://github.com/marckolak/wtController

Copyright (C) 2021 Marcin Kolakowski (CC 4.0)
"""

import re

import numpy as np
import pandas as pd

from src.scan_processing import scan2xy


def load_motion_file(filepath):
    """ Loads motion file

    Parameters
    ----------
    filepath: str
        path to file

    Returns
    -------
    mot: DataFrame
        history of the robot's motion in a DataFrame
    """

    # start_time = datetime.datetime.strptime('2021-01-27 17:54:31.00', '%Y-%m-%d %H:%M:%S.%f').timestamp() + 1

    # get motion history
    mot = pd.read_csv(filepath, names=['ts', 'dir', 'speed']).iloc[1:].reset_index(
        drop=True)

    # mot['ts'] = mot['ts']  # normalize the timestamp to the start_time
    mot = pd.concat([pd.DataFrame({'ts': mot['ts'].min() - 500, 'dir': ['stop']}), mot],
                    ignore_index=True)  # add stop at the beggining

    mot['ts_start'] = mot['ts']
    mot['ts_end'] = np.r_[mot['ts'].values[1:], mot['ts'].max() + 20]

    return mot


def load_scans(filepath, d_limit=(0.2, 15), min_size=50):
    """Loads scans saved by the wild_thumper robot

    Parameters
    ----------
    filepath: str
        path to file
    d_limit: tuple
        limit for range measurements (min, max) - other values will be removed
    min_size: int
        minimum scan size - incomplete scans will be removed

    Returns
    -------
    scans: list
        list of scans
    scans_ts: ndarray
        timestamps for scans

    """
    # read scan file
    with open(filepath, 'r') as f:
        file = f.read()

    # search for individual scans {ts, angles [], dists []} using regex
    search = re.findall(r'(\d*\.\d*)x(\[(\d*[ x])*\d*\])x(\[(\d*[ x])*\d*\])', file.replace('\n', 'x'))

    # extract scans
    scan_ts = []
    scans = []

    for scan in search[1:]:

        # get angles
        astr = scan[1].replace('x', ' ').replace('   ', ' ')[1:-1]
        a = np.radians(np.fromstring(astr, sep=' ').astype('float') / 1000)

        # get distances
        dstr = scan[3].replace('x', ' ').replace('   ', ' ')[1:-1]
        d = np.fromstring(dstr, sep=' ').astype('float') / 100

        # construct scan table and remove outlier values (very small and very large)
        scan_ad = np.c_[a, d]
        scan_ad = scan_ad[scan_ad[:, 1] > d_limit[0]]
        scan_ad = scan_ad[scan_ad[:, 1] < d_limit[1]]

        if scan_ad.shape[0] > min_size:
            # get ts
            scan_ts.append(float(scan[0]))
            scans.append(scan_ad)

    return scans, np.array(scan_ts)


def get_constraint(t1, t0, motion, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0, control_format=False):
    """ Get constraints between two poses in specific time moments

    Parameters
    ----------
    t1: float
        timestamp of the second pose
    t0: float
        timestamp of the first pose
    motion: DataFrame
        motion DataFrame
    LIN_SPEED: float
        linear speed in m/s
    ROT_SPEED_LEFT: float
        rotation speed in Left direction in rad/s
    ROT_SPEED_RIGHT: float
        rotation speed in Right direction in rad/s
    h0: float
        initial heading in radians
    control_format: bool, optional
        if True, return the values in controls format, default:False

    Returns
    -------
    x: ndarray
        constrains [x,y, rot]
    """
    mot = motion[(motion.ts > t0) & (motion.ts < t1)].reset_index()

    h = h0
    x = np.r_[0, 0, 0].astype('float')
    i = 0
    for i, r in mot.iterrows():
        ts_start = np.maximum(r['ts_start'], t0)
        ts_end = np.minimum(r['ts_end'], t1)

        dt = ts_end - ts_start

        direction = r.dir

        if control_format:
            uv = np.r_[1.0, 0.0]
        else:
            uv = np.r_[np.cos(h), np.sin(h)]

        if direction == 'forward':
            x[:2] += uv * dt * LIN_SPEED

        if direction == 'reverse':
            x[:2] += -dt * uv * LIN_SPEED

        if direction == 'left':
            x[2] += dt * ROT_SPEED_LEFT
            h += dt * ROT_SPEED_LEFT

        if direction == 'right':
            x[2] += - dt * ROT_SPEED_RIGHT
            h += - dt * ROT_SPEED_RIGHT

    return x


def get_controls(t1, t0, motion, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0, control_format=False):
    """ Get control vector between two poses in specific moments

    Parameters
    ----------
    t1: float
        timestamp of the second pose
    t0: float
        timestamp of the first pose
    motion: DataFrame
        motion DataFrame
    LIN_SPEED: float
        linear speed in m/s
    ROT_SPEED_LEFT: float
        rotation speed in Left direction in rad/s
    ROT_SPEED_RIGHT: float
        rotation speed in Right direction in rad/s
    h0: float
        initial heading in radians
    control_format: bool, optional
        if True, return the values in controls format, default:False

    Returns
    -------
    u: ndarray
        control vector [d, rot]
    """
    mot = motion[(motion.ts_end >= t0) & (motion.ts_start <= t1)].reset_index()

    h = h0
    u = np.r_[0, 0].astype('float')
    i = 0
    for i, r in mot.iterrows():
        ts_start = np.maximum(r['ts_start'], t0)
        ts_end = np.minimum(r['ts_end'], t1)

        dt = ts_end - ts_start

        direction = r.dir

        if direction == 'forward':
            u[0] += dt * LIN_SPEED

        if direction == 'reverse':
            u[0] += -dt * LIN_SPEED

        if direction == 'left':
            u[1] += dt * ROT_SPEED_LEFT
            h += dt * ROT_SPEED_LEFT

        if direction == 'right':
            u[1] -= dt * ROT_SPEED_RIGHT
            h -= dt * ROT_SPEED_RIGHT

    return u


def select_static_scans(mot, scans, scans_ts):
    """Select scans taken in a static position

    Parameters
    ----------
    mot: DataFrame
        motion DataFrame
    scans: list
        list of scans
    scans_ts: ndarray
        timestamps for scans

    Returns
    -------
    scans_static: list
        list of stationary scans
    scans_static_ts: ndarray
        timestamps for stationary scans
    """
    # bin scans accorrding to timestamps of motion actions
    mot_bins = mot.ts.values
    mot_bin_names = mot.dir.values.astype('str')
    scan_bins = pd.cut(np.array(scans_ts), bins=mot_bins, retbins=False)  # , labels=mot_bin_names[:-1],)
    scan_bins_labels = pd.cut(np.array(scans_ts), bins=mot_bins, retbins=False, labels=mot_bin_names[:-1],
                              ordered=False)

    # prepare lists for median scans
    med_ts, med_labels, med_scans = ([] for i in range(3))

    for c, label in zip(scan_bins.categories, scan_bins_labels):  # group scans by bin

        si = np.argwhere(scan_bins == c)
        if si.size:
            if si.size > 1:
                t = np.min(scans_ts[si])
            else:
                t = (scans_ts[si[0][0]])

            med_ts.append(t)
            med_labels.append(label)

            if si.size > 1:
                scan_full = [scans[i[0]] for i in si]
                scan_full = scan_full[1:]
            else:
                scan_full = scans[si[0][0]]

            scan_full = pd.DataFrame(np.vstack(scan_full), columns=['a', 'd'])
            med_sda = scan_full.groupby('a').min().reset_index()
            med_scans.append(np.c_[scan2xy(scan_full.groupby('a').median().reset_index().values), med_sda['a'].values])

    med_ts = np.array(med_ts)

    med_labels = pd.cut(np.array(med_ts), bins=mot_bins, retbins=False, ordered=False, labels=mot_bin_names[:-1])
    med_labels = np.array(med_labels.tolist())

    scans_static = [med_scans[i] for i in np.argwhere((med_labels == 'stop')).T[0]]
    scans_static_ts = [med_ts[i] for i in np.argwhere((med_labels == 'stop')).T[0]]

    return scans_static, scans_static_ts
