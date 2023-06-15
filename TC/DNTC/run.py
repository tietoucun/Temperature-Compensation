import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.res_plot import res_plot

parser = argparse.ArgumentParser(description='Temperature compensation')
# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='Sample_data', help='model id')
parser.add_argument('--model', type=str, required=False, default='ns_Transformer',
                    help='model name, options: [ns_Transformer, Transformer]')
# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./datas/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Sample_data.csv', help='data file')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--features', type=str, default='M',
                    help='compensation task, options:[M, S]; M:multivariate compensation multivariate, S:univariate compensation univariate')
parser.add_argument('--target', type=str, default='x', help='target feature in S')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='output sequence length')
parser.add_argument('--fre_rat', type=int, default=10, help='Frequency reduction ratio')
parser.add_argument('--par_deta', type=int, default=0.01, help='loss par_deta')
# model define
parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
parser.add_argument('--c_out', type=int, default=6, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=False)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--embed', type=str, default='fixed',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=0, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_false', help='use automatic mixed precision training', default=True)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, default=[256, 256], nargs='+', help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
parser.add_argument('--is_inverse', type=bool, default=True,
                        help='whether inverse transform the compensation results')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        torch.cuda.set_device(args.gpu)

print('Args in experiment:')
print(args)
ssetting = '{}_{}_{}'.format(args.model_id, args.model, args.data)
Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_ta{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_dt{}_fr{}_pd{}_lr{}_d{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.target,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.distil,
            args.fre_rat,
            args.par_deta,
            args.learning_rate,
            args.dropout,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting, ssetting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, ssetting)

        res_plot(setting, ssetting, args.is_inverse, args.fre_rat, args.features, args.target)


        torch.cuda.empty_cache()

else:
    ii = 0
    setting = '{}_{}_{}_ft{}_ta{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_dt{}_fr{}_pd{}_lr{}_d{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.target,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.distil,
        args.fre_rat,
        args.par_deta,
        args.learning_rate,
        args.dropout,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    res_plot(setting, ssetting, args.is_inverse, args.fre_rat, args.features, args.target)
    torch.cuda.empty_cache()

