import torch.optim as optim


def get_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate)

    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
