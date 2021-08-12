## This script can be used to load and preprocess scans in the dataset
## The results is a pickle file with two lists:
## * scans in x-y coordinates collected, when the robot is stationary 
## * controls sent to the robot between the consecutive scans
## Author: Marcin Kolakowski

import pickle
import numpy as np
from src.load_files import  load_ionis_file, distribute_packets_ionis, rearrange_timestamps_ble, measurement_array

results_path = 'tests/test_robot_ble.txt'

# load results
m_ble, m_uwb, t0 = load_ionis_file(results_path, normalize_ts=True)


# preprocess BLE
m_ble = distribute_packets_ionis(m_ble)
m_ble = rearrange_timestamps_ble(m_ble, 1, 3)
arr_ble, ids_ble, df_ble = measurement_array(m_ble, 'ble', data_frame=True)  # create measurement array

df_ble['ts']=(df_ble['ts']+t0)*1e9

print(df_ble.head(10))
