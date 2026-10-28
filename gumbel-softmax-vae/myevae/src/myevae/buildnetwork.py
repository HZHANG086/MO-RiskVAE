import torch

def buildNetwork(layers, activation, add_batchNorm=False, dropout=.0, activate_last=True):
    net = []
    for i in range(1, len(layers)):
        net.append(torch.nn.Linear(layers[i-1], layers[i], bias=not add_batchNorm))
        if add_batchNorm:
            net.append(torch.nn.BatchNorm1d(layers[i]))
        is_last = (i == len(layers) - 1)
        if not is_last or activate_last:
            if dropout > .0:
                net.append(torch.nn.Dropout(dropout))
            net.append(activation)
    outnetwork=torch.nn.Sequential(*net)
    return outnetwork.to(torch.float64)

