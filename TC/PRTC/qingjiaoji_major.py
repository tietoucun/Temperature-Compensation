
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import basefun
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt

def temp_compensation(DATA, input_freq, output_freq, T, model_id, ii):

    freq = input_freq*60
    DATA = DATA.apply(pd.to_numeric, errors='coerce')
    DATA = DATA.dropna(axis=0, how='any')
    DATA = DATA.reset_index(drop=True)

    # 3-sigma filter
    for axis in ['x', 'y', 'z', 'temp']:
        DATA = basefun.sigma3(DATA, axis, freq)

    # median filter
    median_put_freq = input_freq/10
    f = int(input_freq/median_put_freq)
    DATA_medial = pd.DataFrame()
    for axis in ['x', 'y', 'z', 'temp']:
        data_kalman_median = []
        for rms_i in range(0, len(DATA), f):
            DATA_2 = DATA[axis].iloc[rms_i:rms_i + f].median()
            data_kalman_median.append(DATA_2)
        DATA_medial[axis] = pd.Series(data_kalman_median)
    DATA = DATA_medial

    # Kalman filter
    for axis in ['x', 'y', 'z', 'temp']:
        dampling = int(DATA[axis].var())
        DATA["flt_{}".format(axis)] = basefun.kalmanfilter(DATA, axis, dampling)


    freq = int(median_put_freq*60)
    DATA['space'] = basefun.space(DATA['flt_temp'], freq)

    DATA_min_max = basefun.min_max(DATA)

    folder_path = './results/' + model_id + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    result = pd.DataFrame()
    for axis in ['x', 'y', 'z']:
        if T == 1:
            axis_mean = int(DATA['flt_{}'.format(axis)].mean())
        elif T == 0:
            if axis == 'x':
                axis_mean = 10
            elif axis == 'y':
                axis_mean = 24
            elif axis == 'z':
                axis_mean = 1008
        Labels = pd.DataFrame()
        Labels['{}_labels'.format(axis)] = pd.DataFrame(axis_mean - DATA["flt_{}".format(axis)])

        DATA_min_max_labels = pd.concat([DATA, DATA_min_max, Labels], axis=1, ignore_index=True)
        DATA_min_max_labels.columns = ['x', 'y', 'z', 'temp', 'flt_x', 'flt_y', 'flt_z', 'flt_temp', 'space',
                                       'flt_temp_min_max', 'space_min_max', '{}_labels'.format(axis)]
        if T == 1:
            nums_test = int(len(Labels)/4)
        elif T == 0:
            nums_test = len(Labels)
        train, train_data, train_labels, test, test_data, test_labels = basefun.train_test(DATA_min_max_labels, axis, nums_test)
        Temp = test['temp'].values

        if axis == 'x':
            degree = 13
        elif axis == 'y':
            degree = 20
        elif axis == 'z':
            degree = 20
        poly_reg = PolynomialFeatures(degree=degree)
        if T == 1:
            X_poly = poly_reg.fit_transform(train_data)
            poly_reg_model = LinearRegression()
            poly_reg_model = poly_reg_model.fit(X_poly, train_labels)
            joblib.dump(poly_reg_model, folder_path + '{}_{}_{}_{}.bit'.format(model_id, axis, degree, ii))
        elif T == 0:
            poly_reg_model = joblib.load(folder_path + '{}_{}_{}_{}.bit'.format(model_id, axis, degree, ii))
        y_poly = poly_reg.fit_transform(test_data)
        labels_predicted = poly_reg_model.predict(y_poly)
        axis_predicted = pd.DataFrame(test['flt_{}'.format(axis)]).values + labels_predicted
        outputs_y = axis_predicted
        fig, ax1 = plt.subplots(figsize=(15, 4), dpi=200)
        ax1.plot(range(0, len(outputs_y) * 10, 10), outputs_y, '.', c='darkorange', label='out', linestyle='')
        ax1.axhline(np.median(outputs_y), ls="--", c="lime", label='median')
        ax2 = ax1.twinx()
        ax2.plot(range(0, len(Temp) * 10, 10), Temp, 'red', label='Temp', alpha=0.4)
        ax2.set_ylim((Temp.min() // 1) - 2, (Temp.max() // 1) + 2)
        if os.path.exists('plots/data'):
            plt.savefig('./plots/data/' + '{}_{}_dim = {}'.format(model_id, ii, axis) + '.png', bbox_inches='tight')
        else:
            os.makedirs('plots/data')
            plt.savefig('./plots/data/' +  '{}_{}_dim = {}'.format(model_id, ii, axis) + '.png', bbox_inches='tight')
        plt.cla()
        plt.close("all")

        M_axis = np.median(axis_predicted)
        SD_axis = np.std(axis_predicted)


        median_put_freq = 50
        f = int(median_put_freq / output_freq)
        Predicted = pd.DataFrame(map(lambda x: x[0], axis_predicted))
        data_kalman_median = []
        for rms_i in range(0, len(Predicted), f):
            DATA_2 = Predicted.iloc[rms_i:rms_i + f].median()
            data_kalman_median.append(DATA_2)
        m_axis = np.median(data_kalman_median)
        sd_axis = np.std(data_kalman_median)

        f = open("result.txt", 'a')
        f.write('{}_{}_{}_{}'.format(model_id, axis, degree, ii) + "  \n")
        f.write('{} \n{}'
                .format(SD_axis, sd_axis))
        f.write('\n')
        f.write('\n')
        f.write('{} \n{}'
                .format(M_axis, m_axis))
        f.write('\n')
        f.write('\n')
        f.close()
        result[axis] = pd.Series(data_kalman_median)
    result.to_csv(folder_path + '{}_{}.csv'.format(model_id, ii), encoding='utf_8_sig', index=False, header=None)




if __name__=='__main__':
    model_id_list = ['Sample_data']
    for model_id in model_id_list:
        data_path = './data/{}.csv'.format(model_id)
        DATA = pd.read_csv(data_path, usecols=[0, 1, 2, 3],
                           names=['x', 'y', 'z', 'temp'])
        input_freq = 500
        output_freq = 5
        T = 1
        for ii in range(1):
            print('Train-----{}-{}'.format(model_id, ii))
            result = temp_compensation(DATA, input_freq, output_freq, T, model_id, ii)


