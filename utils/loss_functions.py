from torch import nn


def get_loss_function(loss_config, num_classes=None):
    if isinstance(loss_config, str):
        loss_config = {'name': loss_config, 'params': {}}

    loss_name = loss_config['name']
    params = loss_config.get('params', {})

    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()

    elif loss_name == 'binary_cross_entropy':
        return nn.BCEWithLogitsLoss()

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
