import torch.nn as nn


def create_layers(num_layers=1, dim_layers=[256], layer='Linear',
                  activation='ReLU', last_activation='ReLU', softmax_dim=None):

    if not len(dim_layers) == num_layers+1:
        raise Exception('dim_layers has to be same with {}: num_layers+1'.
                        format(num_layers+1))

    layers = []
    for i in range(num_layers):
        if 'Linear' == layer:
            layers.append(nn.Linear(dim_layers[i], dim_layers[i+1]))
        else:
            raise Exception('Need to be fixed!')

        if num_layers-1 == i:
            if 'ReLU' == last_activation:
                layers.append(nn.ReLU())
            elif 'Sigmoid' == last_activation:
                layers.append(nn.Sigmoid())
            elif 'Softmax' == last_activation:
                layers.append(nn.LogSoftmax(dim=softmax_dim))
            else:
                raise Exception('Fix this!')
            return layers

        if 'ReLU' == activation:
            layers.append(nn.ReLU())
        else:
            raise Exception('Need to be fixed!')

    return layers
