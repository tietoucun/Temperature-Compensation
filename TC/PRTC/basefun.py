import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import statistics
def sigma3(DATA, axis, freq):
    data = pd.DataFrame(columns=['date','x', 'y', 'z', 'temp'])
    for i in range(0, len(DATA), freq):
        data_i = DATA.iloc[i:i + freq]
        sigma = data_i[axis].std()
        mu = data_i[axis].mean()
        lw = mu - 3*sigma
        up = mu + 3*sigma
        data_i = data_i[(data_i[axis] > lw) & (data_i[axis] < up)]
        data = pd.concat([data, data_i], ignore_index=True)
    return data
def kalmanfilter(DATA, axis, damping=1):

    data_axis = np.array(pd.DataFrame(DATA[axis]))
    observations = data_axis
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1

    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state
def space(data, freq):
    Q = pd.Series([statistics.mean(data[i:i + freq]) for i in range(0, len(data), freq)])
    V = Q.diff(periods=1)
    V[0] = 0
    space = pd.DataFrame(np.repeat(V.values, freq)[:, np.newaxis], columns=['space'])
    diff_len = len(data) - len(space)
    if diff_len > 0:
        ddf = pd.DataFrame(np.repeat(V.iloc[-1], diff_len)[:, np.newaxis], columns=['space'])
        space = pd.concat([space, ddf], ignore_index=True)
    return space
def min_max(data):
    temp_min = data['flt_temp'].min()
    temp_max = data['flt_temp'].max() + float("1e-8")
    space_min = data['space'].min()
    space_max = data['space'].max() + float("1e-8")
    data_min_max = pd.DataFrame(data, columns=['flt_temp', 'space'])
    data_min_max['flt_temp'] = data_min_max['flt_temp'].apply(lambda x: (x - temp_min)/(temp_max-temp_min))
    data_min_max['space'] = data_min_max['space'].apply(lambda x: (x - space_min)/(space_max-space_min))
    return data_min_max
def train_test(data_min_max_labels, axis, nums_test=0):

    test = data_min_max_labels.sample(n=nums_test, replace=False, random_state=1)
    train = pd.concat([data_min_max_labels, test]).drop_duplicates(keep=False)


    train_data = np.array(pd.DataFrame(train, columns=['flt_temp_min_max', 'space_min_max']))
    train_labels = np.array(pd.DataFrame(train, columns=['{}_labels'.format(axis)]))

    test_data = np.array(pd.DataFrame(test, columns=['flt_temp_min_max', 'space_min_max']))
    test_labels = np.array(pd.DataFrame(test, columns=['{}_labels'.format(axis)]))
    return train, train_data, train_labels, test, test_data, test_labels
