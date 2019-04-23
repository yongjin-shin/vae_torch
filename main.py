from model import *
from runner import *

data_dir = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runner = Runner(device=device)
runner.get_data(data_path=data_dir)

num_epochs = 500
batch_size = 128
lr = 1e-3
x_dim = 784
y_dim = 10
z_dim = 2
encoder_xh_dim, encoder_wh_dim = [512, 512], [512, 512]
decoder_hx_dim, decoder_hw_dim = [512, 512, 512], [512, 512, 512]
ts = time.strftime("%Y%m%d-%H%M%S")

# CVAE
cvae = CVAE(x_dim, y_dim, z_dim, encoder_xh_dim, decoder_hx_dim).to(device)
runner.set_save_dir(sample_path='./samples/cvae-{}'.format(ts), ts=ts)
runner.train(model=cvae,
             optim='adam',
             num_epoch=num_epochs,
             batch_size=batch_size,
             learning_rate=lr,
             save_samples=True,
             save_reconstructions=True)
runner.plot_mean('./samples/cvae_mnist_2d_{}.png'.format(ts))
del cvae
exit()

# JMVAE
jmvae = JMVAE(x_dim, y_dim, z_dim, encoder_xh_dim, encoder_wh_dim, decoder_hx_dim, decoder_hw_dim).to(device)
runner.set_save_dir(sample_path='./samples/jmvae-{}'.format(ts), ts=ts)
runner.train(model=jmvae,
             optim='adam',
             num_epoch=num_epochs,
             batch_size=batch_size,
             learning_rate=lr,
             save_samples=True,
             save_reconstructions=True)
runner.plot_mean('./samples/jmvae_mnist_2d_{}.png'.format(ts))
del jmvae

# VAE
vae = VAE(x_dim, z_dim, encoder_xh_dim, decoder_hx_dim).to(device)
runner.set_save_dir(sample_path='./samples/vae-{}'.format(ts), ts=ts)
runner.train(model=vae,
             optim='adam',
             num_epoch=num_epochs,
             batch_size=batch_size,
             learning_rate=lr,
             save_samples=True,
             save_reconstructions=True)
runner.plot_mean('./samples/vae_mnist_2d_{}.png'.format(ts))
del vae

