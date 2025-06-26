from .cnn_lstm import CNNLSTM
from .cnn_tcn import CNN_TCN
from .ms_tcn import MS_TCN


def get_model(name, **kwargs):
    if name == "cnn_lstm":
        return CNNLSTM(**kwargs)
    elif name == "cnn_tcn":
        return CNN_TCN(**kwargs)
    elif name == "ms_tcn":
        return MS_TCN(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
