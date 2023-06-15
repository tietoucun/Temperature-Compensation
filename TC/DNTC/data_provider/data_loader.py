import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import statistics
warnings.filterwarnings('ignore')

def sigma3(DATA, axis, freq):
    data = pd.DataFrame(columns=['date', 'x', 'y', 'z', 'temp'])
    for i in range(0, len(DATA), freq):
        data_i = DATA.iloc[i:i + freq]
        sigma = data_i[axis].std()
        mu = data_i[axis].mean()
        lw = mu - 3 * sigma
        up = mu + 3 * sigma
        data_i = data_i[(data_i[axis] > lw) & (data_i[axis] < up)]
        data = pd.concat([data, data_i], ignore_index=True)
    return data

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

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                  data_path = 'test1_3300.csv', features='M', target='x',
                  scale=True, timeenc=0, freq='h', ssetting='ssetting', fre_rat=10):

        self.fre_rat = fre_rat
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.ssetting = ssetting

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), usecols=[1, 2, 3, 4], names=['x', 'y', 'z', 'temp'],
                             dtype={"user_id": int, "username": object})
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
        df_raw = df_raw.dropna(axis=0, how='any')
        df_raw = df_raw.reset_index(drop=True)
        cols = list(df_raw.columns)
        datelist = pd.date_range('2022/08/23', periods=len(df_raw), freq='2ms', name='data')
        df_raw['date'] = pd.to_datetime(datelist)
        df_raw = df_raw[['date'] + cols]

        num_train = int(len(df_raw) * 0.6)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len * self.fre_rat, len(df_raw) - num_test - self.seq_len * self.fre_rat]  # 开始节点
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # 3-sigma filter
        for axis in ['x', 'y', 'z', 'temp']:
            df_raw = sigma3(df_raw, axis, 500 * 60)

        folder_path = './results/' + self.ssetting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        median_put_freq = 500
        freq = int(median_put_freq * 60)
        df_raw['space'] = space(df_raw['temp'], freq)

        # Pseudo label
        cols_data = df_raw.columns[1:]
        df_data = df_raw.loc[:, cols_data]
        cols_df = list(df_data.columns[:-2])
        all_mean = df_data.loc[border1s[0]:border2s[0], cols_df].median(axis=0)
        df_data[['l_x', 'l_y', 'l_z']] = all_mean.values - df_data[cols_df].values
        selected_cols = ['l_x', 'l_y', 'l_z', 'temp', 'space'] + cols_df
        df_data = df_data.loc[:, selected_cols]

        if self.set_type == 2:
            raw_data = np.array(df_raw[['x', 'y', 'z']][border1: border2])
            np.save(folder_path + 'raw_testdata.npy', raw_data)
            Temp_testdata = np.array(df_raw[['temp']][border1: border2])
            np.save(folder_path + 'Temp_testdata.npy', Temp_testdata)

        if self.features == 'M':
            df_data = df_data
        elif self.features == 'S':
            df_data = df_data[['l_{}'.format(self.target), 'temp', 'space', '{}'.format(self.target)]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # TT-embed
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['h'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['min'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['s'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


        df_temp = df_data[['temp']][border1: border2].values
        temp_min = -60
        temp_max = 90
        length = 10
        intervals = np.arange(temp_min, temp_max, length)
        data_temp = np.digitize(df_temp, intervals, right=True).reshape(-1, 1)
        data_temp = np.concatenate((data_stamp, data_temp), axis=1)

        self.data_x = data[border1: border2]
        self.data_y = data[border1: border2]
        self.data_stamp = data_temp
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len * self.fre_rat
        r_begin = s_end - self.label_len * self.fre_rat
        r_end = r_begin + self.label_len * self.fre_rat + self.pred_len * self.fre_rat

        seq_x = self.data_x[s_begin:s_end]  # enc_inp
        seq_y = self.data_y[r_begin:r_end]  # dec_inp
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len * self.fre_rat - self.pred_len * self.fre_rat + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

