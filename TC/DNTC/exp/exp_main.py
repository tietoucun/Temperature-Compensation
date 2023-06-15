from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ns_Transformer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
from utils.tools import count_param
import os
import time
import warnings
import numpy as np
from utils.res_plot import loss_plot, Loss_plot
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
torch.set_default_tensor_type(torch.FloatTensor)

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.bs = args.batch_size
        self.dim = args.enc_in

    def _build_model(self):
        model_dict = {
            'ns_Transformer': ns_Transformer,
            'Transformer': Transformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print('INFO: Trainable parameter count: {:.2f}M'.format(count_param(model) / 1000000.0))
        # print('parameters num:{}'.format(count_param(model)))
        return model

    def _get_data(self, flag, ssetting):
        data_set, data_loader = data_provider(self.args, flag, ssetting)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()


        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        Lowfre_Timev = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
                        else:
                            outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
                    else:
                        outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                if self.args.features == 'M':
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_cy[:, -self.args.pred_len:, :3].to(self.device)
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    pred_var = torch.sqrt(torch.var(outputs[:, :, -3:], dim=1, keepdim=True, unbiased=False) + 1e-5).cpu()
                    zer_var = torch.zeros_like(pred_var).float()
                    loss = criterion(pred_var, zer_var) + self.args.par_deta * criterion(pred[:, :, :3], true)
                elif self.args.features == 'S':
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_cy[:, -self.args.pred_len:, 0].to(self.device)
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    pred_var = torch.sqrt(torch.var(pred[:, :, -1].unsqueeze(2), dim=1, keepdim=True, unbiased=False) + 1e-5).cpu()
                    zer_var = torch.zeros_like(pred_var).float()
                    loss = criterion(pred_var, zer_var) + self.args.par_deta * criterion(pred[:, :, 0], true)

                total_loss.append(loss)
                Lowfre_Timev.append(lowfre_time)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, sum(Lowfre_Timev)

    def train(self, setting, ssetting):
        train_data, train_loader = self._get_data(flag='train', ssetting = ssetting)
        vali_data, vali_loader = self._get_data(flag='val', ssetting = ssetting)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now1 = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()  # Adam
        criterion = self._select_criterion()  # MSELoss

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # writer = SummaryWriter("tensorboard_logs")
        epoch_bar = tqdm(range(self.args.train_epochs))
        # tensorboard_path = 'tensorboard_logs/' + setting + '/epoch_loss'
        # if not os.path.exists(tensorboard_path):
        #     os.makedirs(tensorboard_path)

        train_Loss = []
        Lowfre_Time = []
        for epoch in epoch_bar:
            iter_count = 0
            train_loss = []
            Lowfre_time = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
                        else:
                            outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                        if self.args.features == 'M':
                            outputs = outputs[:, -self.args.pred_len:, :]
                            batch_y = batch_cy[:, -self.args.pred_len:, :3].to(self.device)
                            pred_var = torch.sqrt(
                                torch.var(outputs[:, :, -3:], dim=1, keepdim=True, unbiased=False) + 1e-5).to(torch.float32).cpu()
                            zer_var = torch.zeros_like(pred_var).float()
                            loss = criterion(pred_var, zer_var) + self.args.par_deta * criterion(outputs[:, :, :3],
                                                                                                 batch_y)
                        elif self.args.features == 'S':
                            outputs = outputs[:, -self.args.pred_len:, :]
                            batch_y = batch_cy[:, -self.args.pred_len:, 0].to(self.device)
                            pred_var = torch.sqrt(
                                torch.var(outputs[:, :, -1].unsqueeze(2), dim=1, keepdim=True, unbiased=False) + 1e-5).to(torch.float32).cpu()
                            zer_var = torch.zeros_like(pred_var).float()
                            loss = criterion(pred_var, zer_var) + self.args.par_deta * criterion(outputs[:, :, 0],
                                                                                                 batch_y)
                        train_loss.append(loss.item())
                        Lowfre_time.append(lowfre_time)
                else:
                    if self.args.output_attention:
                        outputs, batch_cy, lowfre_time= self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
                    else:
                        outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, batch_y)

                    if self.args.features == 'M':
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_cy[:, -self.args.pred_len:, :3].to(self.device)
                        pred_var = torch.sqrt(
                            torch.var(outputs[:, :, -3:], dim=1, keepdim=True, unbiased=False) + 1e-5).cpu()
                        zer_var = torch.zeros_like(pred_var).float()
                        loss = criterion(pred_var, zer_var) + self.args.par_deta * criterion(outputs[:, :, :3], batch_y)
                    elif self.args.features == 'S':
                        outputs = outputs[:, -self.args.pred_len:, :]
                        batch_y = batch_cy[:, -self.args.pred_len:, 0].to(self.device)
                        pred_var = torch.sqrt(
                            torch.var(outputs[:, :, -1].unsqueeze(2), dim=1, keepdim=True, unbiased=False) + 1e-5).cpu()
                        zer_var = torch.zeros_like(pred_var).float()
                        loss = criterion(pred_var, zer_var) + self.args.par_deta * criterion(outputs[:, :, 0], batch_y)

                    train_loss.append(loss.item())
                    Lowfre_time.append(lowfre_time)  # pool-time
                if (i + 1) % 100 == 0:

                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # loss_plot(setting, train_loss, epoch)
            train_loss = np.average(train_loss)
            vali_loss, fre_timev = self.vali(vali_data, vali_loader, criterion)
            train_Loss.append(train_loss.item())
            Lowfre_Time.append(sum(Lowfre_time))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # writer.close()
        # Loss_plot(setting, train_Loss,)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # tensorboard.close()
        time_cost = time.time() - time_now1
        print('time_cost:{}, Lowfre_Time:{}'.format(time_cost, sum(Lowfre_Time)))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('time_cost:{}, Lowfre_Time:{}'.format(time_cost, sum(Lowfre_Time)))
        f.write('\n')
        f.write('\n')
        f.close()

        return self.model

    def test(self, setting, ssetting, test=0):
        test_data, test_loader = self._get_data(flag='test', ssetting = ssetting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds_l = []
        cel_y = []

        self.model.eval()
        with torch.no_grad():

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # cel_y = batch_y[:, :, self.args.enc_in:].float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]
                        else:
                            outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)[0]

                    else:
                        outputs, batch_cy, lowfre_time = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                assert self.args.target in ['x', 'y', 'z']
                types_map = {'x': 0, 'y': 1, 'z': 2}
                set_types = types_map[self.args.target]
                if self.args.features == 'M':
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_cy[:, -self.args.pred_len:, -3:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                elif self.args.features == 'S':
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_cy[:, -self.args.pred_len:, -1].unsqueeze(2).to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                else:
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_cy[:, -self.args.pred_len:, set_types + 5].unsqueeze(2).to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                cel = batch_y
                preds_l.append(pred)
                cel_y.append(cel)



        preds_l = np.array(preds_l)  # (i,bs,p_len,dims)
        cel_y = np.array(cel_y)
        # inputx = np.array(inputx)
        print('test shape:', preds_l.shape)
        preds_l = preds_l.reshape(-1, preds_l.shape[-2], preds_l.shape[-1])  # (i,bs,p_len,dims) -> (i*bs, p_len, dims)
        cel_y = cel_y.reshape(-1, cel_y.shape[-2], cel_y.shape[-1])
        print('test shape:', preds_l.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.args.is_inverse:

            if self.args.features == 'M':
                m = np.delete(test_data.scaler.mean_, [3, 4])
                v = np.delete(test_data.scaler.var_, [3, 4])
                preds_l = preds_l * (v ** 0.5) + m

                m2 = test_data.scaler.mean_[-3:]
                v2 = test_data.scaler.var_[-3:]
                cel_y = cel_y * (v2 ** 0.5) + m2
            elif self.args.features == 'S':
                 m = np.delete(test_data.scaler.mean_, [1, 2])
                 v = np.delete(test_data.scaler.var_, [1, 2])
                 preds_l = preds_l * (v ** 0.5) + m

                 m2 = test_data.scaler.mean_[-1]
                 v2 = test_data.scaler.var_[-1]
                 cel_y = cel_y * (v2 ** 0.5) + m2


            np.save(folder_path + 'pred_l_inverse.npy', preds_l)
            np.save(folder_path + 'cel_y_inverse.npy', cel_y)
        else:

            np.save(folder_path + 'pred.npy', preds_l)
            np.save(folder_path + 'true.npy', cel_y)


        return
