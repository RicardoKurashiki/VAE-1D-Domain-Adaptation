from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR


def get_scheduler(scheduler_config, optimizer):
    name = scheduler_config.get('name', 'step_lr')
    params = scheduler_config.get('params', {})

    if name == 'none':
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    elif name == 'step_lr':
        step_size = params.get('step_size', 10)
        gamma = params.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif name == 'reduce_on_plateau':
        patience = params.get('patience', 10)
        factor = params.get('factor', 0.1)
        return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)

    else:
        raise ValueError(f"Unknown scheduler: {name}")
