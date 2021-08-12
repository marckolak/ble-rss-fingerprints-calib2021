"""
The pypositioning.system.load_files.py module contains functions allowing to load measurement results from various
types of files. The currently available functions allow to load **.psd** files collected with TI Packet Sniffer and
results obtained using IONIS localization system.

Copyright (C) 2020 Marcin Kolakowski
"""
import numpy as np
import pandas as pd


def load_ionis_file(filepath, normalize_ts=False):
    """ Load measurement file from IONIS system

    Parameters
    ----------
    filepath: str
        path to measurement file
    normalize_ts: bool
        if True set ts base to 0 (rounds ts to full seconds)

    Returns
    -------
    ble_df: DataFrame
        data frame with ble rssi results (contains whole ble packets received by the anchors)
    uwb_df: Dataframe
        data frame with uwb-based toa results
    ts_0: float
        timestamp of the first received packet
    """

    # open psd file
    f = open(filepath, 'r')

    ble_res = []
    uwb_res = []

    # split and decode each line
    for line in f:
        s = line.split('\t')

        # if int(s[3]) == 19:
        #     ...  # the packet is empty
        if int(s[5]) * 12 + int(s[6]) * 8 + 19 > int(s[3]):
            print("faulty packet, ts: " + s[0])
        else:
            # get ble and uwb packets number
            ble_n = int(s[5])
            uwb_n = int(s[6])

            # for each ble packet
            for k in range(ble_n):
                bps = 6 + k * 8
                # append array [ts, an_id, an_sqn, an_pressure, BLE packet contents]
                ble_res.append(s[:3] + [s[4]] + s[bps + 1:bps + 9])

            for j in range(uwb_n):
                ups = 6 + ble_n * 8 + j * 4
                # append array [ts, an_id, an_sqn, an_pressure, BLE packet contents]
                uwb_res.append(s[:3] + s[ups + 1:ups + 5])

    # reshape the arrays
    ble_res = np.array(ble_res)
    uwb_res = np.array(uwb_res)

    if ble_res.size > 0:
        ble_df = pd.DataFrame(data=ble_res,
                              columns=['ts', 'an_id', 'an_sqn', 'an_p', 'rx_id', 'tag_id', 'ble_ts', 'rssi',
                                       'pres', 'volt', 'steps', 'alert'])
        ble_df = ble_df.astype(dtype={'ts': 'float', 'an_id': 'int32', 'an_sqn': 'int32', 'an_p': 'int32',
                                      'rx_id': 'int32', 'tag_id': 'int32', 'ble_ts': 'int32',
                                      'rssi': 'float', 'pres': 'int32', 'volt': 'int32',
                                      'steps': 'int32', 'alert': 'int32'})

        ble_df.loc[ble_df['rssi'] == 0, 'rssi'] = np.nan
    else:
        ble_df = None

    if uwb_res.size > 0:
        uwb_df = pd.DataFrame(data=uwb_res, columns=['ts', 'an_id', 'an_sqn', 'rx_id', 'tag_id', 'uwb_sqn', 'toa'])

        uwb_df = uwb_df.astype({'ts': 'float', 'an_id': 'int32', 'an_sqn': 'int32',
                                'rx_id': 'int32', 'tag_id': 'int32', 'uwb_sqn': 'int32', 'toa': 'float'})
        uwb_df['toa'] = uwb_df['toa'].values * 15.65e-12
    else:
        uwb_df = None

    if normalize_ts:
        ts_min = 0
        if (uwb_res.size > 0 and ble_res.size > 0):
            ts_min = np.minimum(ble_df.ts.min(), uwb_df.ts.min())
            ble_df.ts = np.rint((ble_df.ts - ts_min).values / 1000)
            uwb_df.ts = np.rint((uwb_df.ts - ts_min).values / 1000)

        elif uwb_res.size > 0:
            ts_min = uwb_df.ts.min()
            uwb_df.ts = np.rint((uwb_df.ts - ts_min).values / 1000)
            print('no ble results in a file - normalizing uwb ts only')

        elif ble_res.size > 0:
            ts_min = ble_df.ts.min()
            ble_df.ts = np.rint((ble_df.ts - ts_min).values / 1000)
            print('no uwb results in a file - normalizing ble ts only')

        return ble_df, uwb_df, ts_min / 1000

    return ble_df, uwb_df, 0


def synchronize_toa_ionis(m_uwb, an, an_r):
    """ Synchronize toa values according to IONIS synchronization scheme

    Parameters
    ----------
    m_uwb: DataFrame
        data frame with uwb measurement results
    an: ndarray
        anchor nodes coordinates [id,x,y,z]
    an_r: ndarray
        reference anchor node coordinates [x,y,z]

    Returns
    -------
    m_uwb: DataFrame
        m_uwb data frame with toa values synchronized
    """
    # initialize array with empty rows for missing anchors
    an_f = np.empty((int(an[:, 0].max()), 3))
    an_f[:] = np.NaN
    for a in an:
        an_f[int(a[0]) - 1, :] = a[1:]

    m_uwb["toa"] = m_uwb.toa + np.linalg.norm(an_f[m_uwb.an_id - 1] - an_r, axis=1) / 3e8

    return m_uwb


def distribute_packets_ionis(df):
    """ Distribute packets that could be delayed and came to the system controller at the same time

    Parameters
    ----------
    df: DataFrame
        dataframe containing measurement results with timestamps and sqns. It must include columns:
        [ts, an_sqn, an_id]. Timestamp must be rounded to full seconds (might be float)

    Returns
    -------
    df_d: DataFrame
        dataframe, where the packet ts were corrected to the reception times, which would occur
        without the delay
    """

    # copy the dataframe
    df_d = df.copy()

    # get unique anchor ids
    anchor_ids = df.an_id.unique()

    # for each anchor search for delayed packets and distribute them
    for an_id in anchor_ids:
        mask_an_id = df.an_id == an_id
        uts = df[mask_an_id].ts.unique()

        for i in range(uts.size):
            ts = df[mask_an_id & (df.ts == uts[i])]
            an_sqns = ts.an_sqn.unique()
            # if the results
            if an_sqns.size > 1:
                # find last properly received packet
                pi = 1
                while df[mask_an_id & (df.ts == uts[i - pi])].an_sqn.unique().size > 1:
                    pi = pi + 1
                prev_ts = uts[i - pi]
                prev_an_sqn = df_d[(df_d.an_id == an_id) & (df_d.ts == uts[i - pi])].an_sqn.values[0]

                # correct timestamps
                tse = distribute_packet_batch(ts.ts, ts.an_sqn, prev_ts, prev_an_sqn)
                df_d.ts[(df_d.an_id == an_id) & (df_d.ts == uts[i])] = tse

    return df_d


def distribute_packet_batch(ts, an_sqn, ts_p, an_sqn_p):
    """Correct timestamps of the packets, which were received in a batch due to delay introduced in the WiFi interface.

    Parameters
    ----------
    ts: array_like
        timestamps of packets received in a batch [in seconds]
    an_sqn: array_like
        anchor sqns of packets received in a batch [0-255]
    ts_p: float
        the timestamp of the last properly received packet
    an_sqn_p: int
        the anchor sqn of the last properly received packet

    Returns
    -------
    tse: ndarray
        timestamps corrected to the reception times, which would occur without the delay
    """

    # empty list for collected packets
    tse = []

    for t, ans in zip(ts, an_sqn):
        # check if anchor sqn is higher than the previous one or the counter has turned
        if ans >= an_sqn_p:
            te = ts_p + ans - an_sqn_p
        else:
            te = ts_p + (256 + ans - an_sqn_p)

        tse.append(te)

    return np.array(tse)




def rearrange_timestamps_ble(m_ble, tag_id, packet_rate, distribute_delayed=False):
    """Change timestamps values so that the consecutive packets sent at different times do not have the same ts.

    The timestamps are changed as follows: \
        new_ts = ts + 1/packet_rate * N \
    where N is the sequential number of BLE packet inside WiFi frame.
    
    Parameters
    ----------
    m_ble: DataFrame
        dataframe with measurement results
    tag_id: int
        tag identifier
    packet_rate: float
        packet rate set in the systems [packets per second]
    distribute_delayed: bool, optional
        if True call distribute_packets_ionis

    Returns
    -------
    m_b: DataFrame
        Input m_ble DataFrame with rearranged timestamps.
    """

    # filter tag id
    m_b = m_ble[m_ble.tag_id == tag_id]

    if distribute_delayed:  # distribute delayed packets
        m_b = distribute_packets_ionis(m_b)

    # group and bin by BLE ts
    grouped = m_b.groupby(by=['ts', 'an_id', 'an_sqn', 'tag_id'])
    bins = []
    for n, g in grouped:
        bins.append(pd.cut(g.ble_ts, packet_rate, labels=False))
    m_b['bin'] = pd.concat(bins)

    # group and average power per BLE receiver
    grouped = m_b.groupby(by=['ts', 'an_id', 'tag_id', 'bin'])
    m_b = grouped.agg({'rssi': log_mean})
    m_b = m_b.reset_index()

    # get ts with 1/rate
    m_b['ts'] = m_b['ts'] + m_b['bin'] / packet_rate

    return m_b


def rearrange_timestamps_uwb(m_uwb, tag_id, packet_rate, distribute_delayed=False):
    """Change timestamps values so that the consecutive packets sent at different times do not have the same ts.

    The timestamps are changed as follows: \
        new_ts = ts + 1/packet_rate * N \
    where N is the sequential number of UWB packet inside WiFi frame.

    Parameters
    ----------
    m_uwb: DataFrame
        dataframe with measurement results
    tag_id: int
        tag identifier
    packet_rate: float
        packet rate set in the systems [packets per second]
    distribute_delayed: bool, optional
        if True call distribute_packets_ionis

    Returns
    -------
    m_u: DataFrame
        Input m_uwb DataFrame with rearranged timestamps.
    """

    # filter tag id
    m_u = m_uwb[m_uwb.tag_id == tag_id].copy()

    if distribute_delayed:  # distribute delayed packets
        m_u = distribute_packets_ionis(m_u)

    # group and bin by reception ts (in this case toa value)
    grouped = m_u.groupby(by=['ts', 'an_sqn'])
    bins = []
    for n, g in grouped:
        bins.append(pd.cut(g.toa, packet_rate, labels=False))
        
        
    m_u['bin'] = pd.concat(bins)

    # get ts with 1/rate
    m_u['ts'] = m_u['ts'] + m_u['bin'] / packet_rate

    return m_u


def measurement_array(m_df, mtype, data_frame=False):
    """Create measurement array [ts, meas values...]

    Parameters
    ----------
    m_df: DataFrame
        measurement dataframe
    mtype: str
        measurement type: 'ble', 'uwb'
    data_frame: bool
        return dataframe, None tuple if true

    Returns
    -------
    array: ndarray
        measurement array in format [ts, mx, my, mz ...]
    an_ids: ndarray
        anchor_ids: [x,y,z ...]
    df: DataFrame, optional
        measurement array stored as dataframe (returned when data_frame==True)
    """
    if mtype == 'uwb':
        m = m_df[['ts', 'uwb_sqn', 'toa', 'an_id']].copy()
    elif mtype == 'ble':
        m = m_df[['ts', 'rssi', 'an_id']].copy()
    else:
        print("Unknown type")
        return None, None

    df = None

    # get unique anchor ids
    anchor_ids = np.sort(m.an_id.unique())

    # create array
    if mtype == 'uwb':
        for i in anchor_ids:
            mp = m[m.an_id == i].rename(columns={'toa': 'toa_' + str(i)}).drop(columns='an_id')
            if df is None:
                df = mp
            else:
                df = df.merge(mp, how='outer', on=['ts', 'uwb_sqn'])
        df = df.sort_values(['ts', 'uwb_sqn'], ascending=[True, True]).reset_index(drop=True)
        df = df.drop(columns='uwb_sqn')

    elif mtype == 'ble':
        for i in anchor_ids:
            mp = m[m.an_id == i].rename(columns={'rssi': 'rssi_' + str(i)}).drop(columns='an_id')
            if df is None:
                df = mp
            else:
                df = df.merge(mp, how='outer', on=['ts'])
        df = df.sort_values(['ts'], ascending=[True]).reset_index(drop=True)

    array = df.values
    anchor_ids = np.r_[0, anchor_ids] # add 0 for ts column
    if data_frame:
        return array, anchor_ids, df

    return array, anchor_ids


def hybrid_array(dfs, on='ts', how='outer'):
    """ Create a hybrid measurement array

    Parameters
    ----------
    dfs: iterable
        DataFrames which would be merged into a hybrid frame
    on: str, default: 'ts'
        on which column the frames will be merged
    how: str, default: 'outer'
        how the frames will be merged

    Returns
    -------
    m: ndarray
        measurement array in format [ts, results...]
    m_type: ndarray
        type of data in each of the columns e.g. ['ts', 'rssi', 'toa]
    m_id: ndarray
        anchor ids of the columns df. Default id is 0 - for 'ts' and other parameter
        not associated with any particular anchor.
    df: DataFrame
        hybrid DataFrame
    """
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on=on, how=how)

    m_type= np.array([x.split('_')[0] for x in df.columns[:]])
    m_id= np.array([x.split('_')[1] if '_' in x else 0 for x in df.columns[:] ]).astype('int')

    return df.values, m_type, m_id, df


def log_mean(v_db, axis=None):
    """ Calculate average for values in log scale by converting to linear and back to log.

    Parameters
    ----------
    v_db: ndarray
        values in log scale
    axis: {int, None}, optional
        axis along which the mean will be calculated

    Returns
    -------
    avg: {ndarray, double}
        mean value

    """

    v_lin = 10 ** (v_db / 10)  # Power in mW
    l_mean = np.nanmean(v_lin, axis=axis)
    db_mean = 10 * np.log10(l_mean)

    return db_mean
