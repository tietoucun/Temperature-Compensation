import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def res_plot(setting, ssetting, is_inverse, fre_rat, features, target):
    path = './results/' + setting
    path1 = './results/' + ssetting
    if is_inverse:
        pred_l_inverse_npy = np.load(path + '/pred_l_inverse.npy')
        cel_y_inverse_npy = np.load(path + '/cel_y_inverse.npy')
        raw_Data = np.load(path1 + '/raw_testdata.npy')

    else:
        pred_l_inverse_npy = np.load(path + '/pred.npy')
        cel_y_inverse_npy = np.load(path + '/true.npy')
        raw_Data = np.load(path1 + '/raw_testdata.npy')
    Temp = np.load(path1 + '/Temp_testdata.npy')
    pred_first = data_load(pred_l_inverse_npy, fre_rat)
    cel_y_first = data_load(cel_y_inverse_npy, fre_rat)
    assert target in ['x', 'y', 'z']
    types_map = {'x': 0, 'y': 1, 'z': 2}
    set_types = types_map[target]
    if features == 'M':
        outputs_first = pred_first[:, :3] + cel_y_first
        pred_x = pred_first[:, -3:]
        pred_first = pred_first[:, :3]
    else:
        outputs_first = pred_first[:, 0].reshape(-1, 1) + cel_y_first
        pred_x = pred_first[:, -1].reshape(-1, 1)
        pred_first = pred_first[:, 0].reshape(-1, 1)
        raw_Data = raw_Data[:, set_types].reshape(-1, 1)

    sd_out = outputs_first.std(axis=0)
    sd_l = pred_first.std(axis=0)
    sd_pred_x = pred_x.std(axis=0)
    sd_conv = cel_y_first.std(axis=0)
    sd_raw = raw_Data.std(axis=0)

    m_out = np.median(outputs_first, axis=0)
    m_l = np.median(pred_first, axis=0)
    m_pred_x = np.median(pred_x, axis=0)
    m_conv = np.median(cel_y_first, axis=0)
    m_raw = np.median(raw_Data, axis=0)

    print('sd_out:{}, sd_l:{},sd_pred_x:{},sd_conv:{}, sd_raw:{}'
          .format(sd_out, sd_l,sd_pred_x, sd_conv, sd_raw))
    print('m_out:{}, m_l:{},m_pred_x:{},m_conv:{}, m_raw:{}'
            .format(m_out, m_l, m_pred_x, m_conv, m_raw))

    f = open("result.txt", 'a')
    f.write(setting + "  \n")
    f.write('sd_out:{}, sd_pred_x:{} ,sd_l:{},sd_conv:{}, sd_raw:{}'
            .format(sd_out, sd_l, sd_pred_x, sd_conv, sd_raw))
    f.write('\n')
    f.write('\n')
    f.write('m_out:{}, m_l:{},m_pred_x:{},m_conv:{}, m_raw:{}'
            .format(m_out, m_l, m_pred_x, m_conv, m_raw))
    f.write('\n')
    f.write('\n')
    f.close()
    if features == 'M':
        for dim in range(pred_first.shape[-1]):
            raw_y, outputs_y = raw_Data[:, dim], outputs_first[:, dim]
            fig, ax1 = plt.subplots(figsize=(15, 4), dpi=200)
            ax1.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
            ax1.plot(range(len(raw_y)), raw_y, '.', label='Raw_data', linestyle='', markersize=1, alpha=0.6)
            ax1.plot(range(0, len(outputs_y) * fre_rat, fre_rat), outputs_y, '.', c="darkorange", label='Out',
                     linestyle='', markersize=1)
            ax1.axhline(np.median(raw_y), ls="--", c='lime', label='median')
            ax2 = ax1.twinx()
            ax2.plot(range(len(Temp)), Temp, 'red', label='Temp', alpha=0.4)
            ax2.set_ylim((Temp.min() // 1) - 2, (Temp.max() // 1) + 2)
            if os.path.exists('plots/data'):
                plt.savefig('./plots/data/' + setting + '_dim = {0}.pdf'.format(dim), format='pdf',
                            pad_inches=0)
            else:
                os.makedirs('plots/data')
                plt.savefig('./plots/data/' + setting + '_dim = {0}.pdf'.format(dim), format='pdf', pad_inches=0)
            # plt.close()
            plt.clf()
            fig2, ax21 = plt.subplots(figsize=(15, 4), dpi=200)
            ax21.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
            ax21.plot(range(0, len(outputs_y) * fre_rat, fre_rat), outputs_y, '.', c="darkorange", label='Out',
                      linestyle='', markersize=1)
            ax21.axhline(np.median(outputs_y), ls="--", c='lime', label='median')
            ax22 = ax21.twinx()
            ax22.plot(range(len(Temp)), Temp, 'red', label='Temp', alpha=0.4)
            ax22.set_ylim((Temp.min() // 1) - 2, (Temp.max() // 1) + 2)
            ax21.set_xlabel(
                'Std_out:{:.4e}, Std_raw:{:.4e}'.format(sd_out[dim], sd_raw[dim]))
            # fig2.legend(loc='upper left')
            if os.path.exists('plots/data/tc'):
                plt.savefig('./plots/data/tc/' + setting + '_dim = {0}.pdf'.format(dim), format='pdf',
                            pad_inches=0)
            else:
                os.makedirs('plots/data/tc')
                plt.savefig('./plots/data/tc/' + setting + '_dim = {0}.pdf'.format(dim), format='pdf', pad_inches=0)
            plt.clf()
    else:
        dim = set_types
        raw_y,  outputs_y = raw_Data, outputs_first
        fig, ax1 = plt.subplots(figsize=(15, 4), dpi=200)
        ax1.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
        ax1.plot(range(len(raw_y)), raw_y, '.', label='Raw_data', linestyle='', markersize=1, alpha=0.6)
        ax1.plot(range(0, len(outputs_y) * fre_rat, fre_rat), outputs_y, '.', c="darkorange", label='Out', linestyle='',
                 markersize=1)
        ax1.axhline(np.median(raw_y), ls="--", c='lime', label='median')
        ax2 = ax1.twinx()
        ax2.plot(range(len(Temp)), Temp, 'red', label='Temp', alpha=0.4)
        ax2.set_ylim((Temp.min() // 1) - 2, (Temp.max() // 1) + 2)

        ax2.set_ylabel('Temp')
        ax1.set_ylabel('Dim = {}'.format(dim))
        if os.path.exists('plots/data'):
            plt.savefig('./plots/data/' + setting + '_dim = {0}.pdf'.format(dim), format='pdf',
                        pad_inches=0)
        else:
            os.makedirs('plots/data')
            plt.savefig('./plots/data/' + setting + '_dim = {0}.pdf'.format(dim), format='pdf', pad_inches=0)
        # plt.close()
        plt.clf()
        fig2, ax21 = plt.subplots(figsize=(15, 4), dpi=200)
        ax21.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
        ax21.plot(range(0, len(outputs_y) * fre_rat, fre_rat), outputs_y, '.', c="darkorange", label='Out',
                  linestyle='',
                  markersize=1)
        ax21.axhline(np.median(outputs_y), ls="--", c='lime', label='median')
        ax22 = ax21.twinx()
        ax22.plot(range(len(Temp)), Temp, 'red', label='Temp', alpha=0.4)
        ax22.set_ylim((Temp.min() // 1) - 2, (Temp.max() // 1) + 2)
        # fig2.legend(loc='upper left')
        if os.path.exists('plots/data/tc'):
            plt.savefig('./plots/data/tc/' + setting + '_dim = {0}.png'.format(dim),
                        pad_inches=0)
        else:
            os.makedirs('plots/data/tc')
            plt.savefig('./plots/data/tc/' + setting + '_dim = {0}.png'.format(dim))
        plt.clf()
def data_load(data, fre_rat):
    batch_num = data.shape[0]
    pred_len = data.shape[1]
    fea_num = data.shape[2]
    pre_len_list = [i for i in range(0, batch_num, pred_len*fre_rat)]
    first_data = []
    for i in pre_len_list:
        tmp_data = data[i, :, :]
        first_data.append(tmp_data)
    first_data = np.array(first_data).reshape(-1, fea_num)
    return first_data

def loss_plot(setting, loss, epoch):
    plt.figure(figsize=(15, 4))
    plt.plot(range(len(loss)), loss, label='loss')
    plt.title(setting + '_Epoch:{}'.format(epoch))
    plt.xlabel('batch')
    plt.legend()
    # plt.show()

    if os.path.exists('plots/loss'):
        plt.savefig(f'./plots/loss/{setting}_{epoch}.png')
    else:
        os.makedirs('plots/loss')
        plt.savefig(f'./plots/loss/{setting}_{epoch}.png')


def Loss_plot(setting, Loss):
    plt.figure(figsize=(15, 4))
    plt.plot(range(len(Loss)), Loss, label='Loss')
    plt.title(setting)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    if os.path.exists('plots/loss_epoch'):
        plt.savefig('./plots/loss_epoch/' + setting + '.png')
    else:
        os.makedirs('plots/loss_epoch')
        plt.savefig('./plots/loss_epoch/' + setting + '.png')



