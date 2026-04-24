from torch.nn import Module, CrossEntropyLoss, MSELoss, L1Loss


def get_loss_fn(name: str) -> Module:
    if name == "cross_entropy":
        return CrossEntropyLoss()
    elif name == "mse":
        return MSELoss()
    elif name == "l1":
        return L1Loss()
    else:
        raise ValueError("Unknown loss function: {}".format(name))
