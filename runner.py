from data import Dataset
from torch.nn import BCELoss, NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import time


class Runner:
    def __init__(self, device):
        self.device = device
        self.data_path, self.sample_path, self.ts = "", "", ""
        self.train_data, self.test_data = None, None
        self.train_loader, self.test_loader = None, None
        self.input_dim, self.num_label = 0, 0
        self.num_samples = 0
        self.num_batch = 0
        self.anneal_param = 0

        self.model, self.opt = None, None
        self.reconst_loss_x, self.reconst_loss_w = None, None
        self.batch_size = 0
        self.train_loss, self.eval_loss = [], []

    def get_data(self, data_path):
        self.data_path = data_path
        train_path = data_path + '/test.csv'  # train_path = data_path + '/train.csv'
        test_path = data_path + '/test.csv'

        self.train_data = Dataset(train_path)
        self.test_data = Dataset(test_path)

        self.input_dim = self.train_data.x_dim
        self.num_label = self.train_data.num_label
        self.num_samples = self.train_data.__len__()

    def set_save_dir(self, sample_path, ts):
        self.ts = ts
        self.sample_path = sample_path
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)

    def train(self, model, optim, num_epoch, batch_size, learning_rate, save_samples=True, save_reconstructions=True):
        self.model = model
        self.model.train()

        self.batch_size = batch_size
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
        self.num_batch = len(self.train_loader)

        self.opt = self.set_opt(optim, learning_rate)
        self.reconst_loss_x, self.reconst_loss_w = self.set_reconst_loss()

        self.set_weight(num_epoch, self.num_batch)
        weight = 1
        for epoch in range(num_epoch):
            rx_loss, rw_loss, kl_loss, tot_loss = 0, 0, 0, 0
            tic = time.time()
            for i, (x, w, l) in enumerate(self.train_loader):
                x = x.to(device=self.device, dtype=torch.float).view(-1, self.input_dim)
                w = w.to(device=self.device, dtype=torch.float).view(-1, self.num_label)
                in_put = {'x': x, 'w': w}
                output, mean, log_var, z_sample = self.model(in_put)

                loss_kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss_x = self.reconst_loss_x(output['x'], x)
                loss = loss_x + loss_kl

                loss_w = 0
                if 'JMVAE' == self.model.whoami:
                    loss_w = self.reconst_loss_w(output['w'], l)
                    weight = min(1, self.get_weight(epoch, i))
                    loss = (loss_x + loss_w) + weight * loss_kl

                loss = self.num_samples * loss / batch_size
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                rx_loss += loss_x
                rw_loss += loss_w
                kl_loss += loss_kl
                tot_loss += loss

                if i+1 == self.num_batch:
                    print(
                        "Epoch[{}/{}], Loss: {:.4f}, KL Div: {:.4f}, X reconst Loss: {:.4f}, W reconst Loss: {:.4f}, Annealing Param: {:4f}, Time: {:4f}".format(
                            epoch + 1, num_epoch, tot_loss / self.num_batch, kl_loss / self.num_batch,
                            rx_loss / self.num_batch, rw_loss / self.num_batch, weight, time.time()-tic))

                    with torch.no_grad():
                        if save_samples:
                            self.save_s(epoch)

                        if save_reconstructions:
                            self.save_r(in_put, epoch)

            self.train_loss.append(tot_loss/self.num_batch)

    # def eval(self):
    #     # test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True)

    def set_opt(self, optim, learning_rate):
        # Optimizer
        if 'adam' == optim:
            opt = Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise Exception('Fix me!')
        return opt

    def set_reconst_loss(self):
        loss_x, loss_w = None, None
        loss_x = BCELoss(reduction='sum')
        if 'JMVAE' == self.model.whoami:
            loss_w = NLLLoss(reduction='sum')
        return loss_x, loss_w

    def set_weight(self, num_epoch, num_batch):
        self.anneal_param = 1 / (2/5 * num_epoch * num_batch)

    def get_weight(self, epoch, step):
        return self.anneal_param * (epoch * self.num_batch + step)

    def save_s(self, epoch):
        z = torch.randn(10, self.model.z_dim).to(self.device)
        if 'CVAE' == self.model.whoami:
            z = torch.cat(tensors=(z, torch.Tensor(np.identity(10))), dim=1)
        out = self.model.decoder(z)
        save_image(out['x'].view(-1, 1, 28, 28), os.path.join(self.sample_path, 'sampled-{}.png'.format(epoch+1)))

    def save_r(self, in_put, epoch):
        out, _, _, _ = self.model(in_put)
        x_concat = torch.cat((in_put['x'].view(-1, 1, 28, 28), out['x'].view(-1, 1, 28, 28)), dim=3)
        save_image(x_concat, os.path.join(self.sample_path, 'reconst-{}.png'.format(epoch+1)))

        if 'JMVAE' == self.model.whoami:
            f = open("./samples/reconst_w_{}.txt".format(self.ts), "a")
            f.write(" ".join(str(e) for e in np.argmax(in_put['w'], axis=1).detach().tolist()))
            f.write("\n")
            f.write(" ".join(str(e) for e in np.argmax(out['w'], axis=1).detach().tolist()))
            f.write("\n\n")
            f.close()

    def plot_mean(self, path):
        if not 2 == self.model.z_dim:
            raise Exception("Cannot float over 2 dimensions: model has {} dimension".format(self.model.z_dim))

        self.model.eval()
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=3000, shuffle=True)
        data, label = [], []
        for i, (x, w, l) in enumerate(self.test_loader):
            if i == 0:
                x = x.to(device=self.device, dtype=torch.float).view(-1, self.input_dim)
                w = w.to(device=self.device, dtype=torch.float).view(-1, self.num_label)
                in_put = {'x': x, 'w': w}
                output, mean, log_var, _ = self.model(in_put)
                data = mean.detach().numpy()
                label = l.detach().numpy()

        color_iter = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
                      '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
                      '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']

        for i in range(self.num_label):
            idx = np.where(label == i)
            plt.scatter(data[idx, 0], data[idx, 1], color=color_iter[i], label=i)

        plt.legend(loc='best')
        plt.title(self.model.whoami, fontsize=8)
        plt.savefig(path)
        plt.close()
        # plt.show()
