import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import create_layers


class VAE(nn.Module):

    def __init__(self, x_dim, z_dim, encoder_h_dim, decoder_h_dim):
        super(VAE, self).__init__()
        self.whoami = 'VAE'
        self.z_dim = z_dim

        self.hidden2log_var = nn.Linear(encoder_h_dim[-1], z_dim)
        self.hidden2mean = nn.Linear(encoder_h_dim[-1], z_dim)

        self.encode_x2hidden = nn.Sequential(*create_layers(
            num_layers=len(encoder_h_dim),
            dim_layers=[x_dim]+encoder_h_dim,
            layer='Linear',
            activation='ReLU',
            last_activation='ReLU'
        ))

        # Bernoulli
        self.decode_z2x = nn.Sequential(*create_layers(
            num_layers=len(decoder_h_dim + [x_dim]),
            dim_layers=[z_dim] + decoder_h_dim + [x_dim],
            layer='Linear',
            activation='ReLU',
            last_activation='Sigmoid'
        ))

    def encoder(self, in_put):
        if self.whoami == 'CVAE':
            in_put = torch.cat(tensors=(in_put['x'], in_put['w']), dim=1)
        elif self.whoami == 'VAE':
            in_put = in_put['x']

        hidden = self.encode_x2hidden(in_put)
        mean = self.hidden2mean(hidden)
        log_var = self.hidden2log_var(hidden)
        return mean, log_var

    def decoder(self, z_sample):
        output = {'x': self.decode_z2x(z_sample)}
        return output

    def reparametrize(self, mean, log_var):
        eps = torch.randn_like(mean)
        std = torch.exp(log_var/2)
        return mean + torch.mul(eps, std)

    def forward(self, in_put):
        mean, log_var = self.encoder(in_put)
        z_sample = self.reparametrize(mean, log_var)
        if 'CVAE' == self.whoami:
            z_sample = torch.cat(tensors=(z_sample, in_put['w']), dim=1)
        elif 'VAE' == self.whoami:
            z_sample = z_sample
        else:
            raise Exception("Fix me!!!!!!")

        output = self.decoder(z_sample)
        return output, mean, log_var, z_sample


class CVAE(VAE):
    def __init__(self, x_dim, w_dim, z_dim, encoder_h_dim, decoder_h_dim):
        super().__init__(x_dim, z_dim, encoder_h_dim, decoder_h_dim)
        self.whoami = 'CVAE'
        self.encode_x2hidden = nn.Sequential(*create_layers(
            num_layers=len(encoder_h_dim),
            dim_layers=[x_dim + w_dim] + encoder_h_dim,
            layer='Linear',
            activation='ReLU',
            last_activation='ReLU'
        ))

        # Bernoulli
        self.decode_z2x = nn.Sequential(*create_layers(
            num_layers=len(decoder_h_dim + [x_dim]),
            dim_layers=[z_dim + w_dim] + decoder_h_dim + [x_dim],
            layer='Linear',
            activation='ReLU',
            last_activation='Sigmoid'
        ))


class JMVAE(VAE):

    def __init__(self, x_dim, w_dim, z_dim, encoder_hx_dim, encoder_hw_dim,
                 decoder_hx_dim, decoder_hw_dim):
        if not encoder_hx_dim[-1] == encoder_hw_dim[-1]:
            raise Exception('the last dimension of encoders should be same, \
                            but {} : {}'.format(encoder_hx_dim[-1],
                                                encoder_hw_dim[-1]))

        super().__init__(x_dim, z_dim, encoder_hx_dim, decoder_hx_dim)
        self.whoami = 'JMVAE'
        self.h2last_h = nn.Linear(encoder_hx_dim[:-1][-1] + ([w_dim]+encoder_hw_dim)[:-1][-1],
                                  encoder_hx_dim[-1])

        self.encode_x2hidden = nn.Sequential(*create_layers(
            num_layers=len(encoder_hx_dim[:-1]),
            dim_layers=[x_dim] + encoder_hx_dim[:-1],
            layer='Linear',
            activation='ReLU',
            last_activation='ReLU'
        ))

        self.encode_w2hidden = nn.Sequential(*create_layers(
            num_layers=len(encoder_hw_dim[:-1]),
            dim_layers=[w_dim] + encoder_hw_dim[:-1],
            layer='Linear',
            activation='ReLU',
            last_activation='ReLU'
        ))

        # Categorical
        self.decode_z2w = nn.Sequential(*create_layers(
            num_layers=len(decoder_hw_dim + [w_dim]),
            dim_layers=[z_dim] + decoder_hw_dim + [w_dim],
            layer='Linear',
            activation='ReLU',
            last_activation='Softmax',
            softmax_dim=1
        ))

    def encoder(self, in_put):
        hidden_x = self.encode_x2hidden(in_put['x'])
        hidden_w = self.encode_w2hidden(in_put['w'])
        hidden_xw = torch.cat((hidden_x, hidden_w), 1)
        hidden = F.relu(self.h2last_h(hidden_xw))
        mean = self.hidden2mean(hidden)
        log_var = self.hidden2log_var(hidden)
        return mean, log_var

    def decoder(self, z_sample):
        output_x = self.decode_z2x(z_sample)
        output_w = self.decode_z2w(z_sample)
        output = {'x': output_x, 'w': output_w}
        return output

    def forward(self, input):
        _, mean, log_var, z_sample = super().forward(input)
        output = self.decoder(z_sample)
        return output, mean, log_var, z_sample
