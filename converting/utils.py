import torch

def param_conv2d(layer: torch.nn.Conv2d) -> dict:
    
    return {
        'in_channels': layer.in_channels,
        'out_channels': layer.out_channels,
        'kernel_size': layer.kernel_size,
        'stride': layer.stride,
        'padding': layer.padding,
        'bias': layer.bias is not None,
    }

def param_trans2d(layer: torch.nn.ConvTranspose2d) -> dict:
    return {
        'in_channels': layer.in_channels,
        'out_channels': layer.out_channels,
        'kernel_size': layer.kernel_size,
        'stride': layer.stride,
        'padding': layer.padding,
        'output_padding': layer.output_padding,
        'bias': layer.bias is not None,
    }

def param_linear(layer: torch.nn.Linear) -> dict:
    return {
        'in_features': layer.in_features,
        'out_features': layer.out_features,
        'bias': layer.bias is not None,
    }

def param_batchnorm(layer: torch.nn.BatchNorm2d) -> dict:
    return {
        'num_features': layer.num_features,
        'eps': layer.eps,
        'momentum': layer.momentum,
        'affine': layer.affine,
        'track_running_stats': layer.track_running_stats,
    }

class Params:
    def __init__(self, Sequential):
        self.sequential = Sequential
    
    def get_params(self):
        layer_para = {}
        if not isinstance(self.sequential, torch.nn.Sequential):
            if isinstance(self.sequential, torch.nn.Conv2d):
                layer_para['conv2d'] = param_conv2d(self.sequential)
            elif isinstance(self.sequential, torch.nn.ConvTranspose2d):
                layer_para['trans2d'] = param_trans2d(self.sequential)
            elif isinstance(self.sequential, torch.nn.Linear):
                layer_para['linear'] = param_linear(self.sequential)
            elif isinstance(self.sequential, torch.nn.BatchNorm2d):
                layer_para['batchnorm'] = param_batchnorm(self.sequential)
            elif isinstance(self.sequential, torch.nn.ReLU) or isinstance(self.sequential, torch.nn.LeakyReLU):
                layer_para['relu'] = {}
            else:
                raise ValueError(f"Layer {self.sequential} is not supported")
            return layer_para
            
        for layer in self.sequential:
            if isinstance(layer, torch.nn.Conv2d):
                layer_para['conv2d'] = param_conv2d(layer)
            elif isinstance(layer, torch.nn.ConvTranspose2d):
                layer_para['trans2d'] = param_trans2d(layer)
            elif isinstance(layer, torch.nn.Linear):
                layer_para['linear'] = param_linear(layer)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                layer_para['batchnorm'] = param_batchnorm(layer)
            elif isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.LeakyReLU):
                layer_para['relu'] = {}
            else:
                raise ValueError(f"Layer {layer} is not supported")
        return layer_para
    
    def get_last_params(self):
        # 5개의 seq 가지고 있음, trans2는 2개임, 
        # 마지막 trans2d, batchnorm, relu, trans2d, tanh임
        layer_para = {}
        trans_count = 0
        for layer in self.sequential:
            if isinstance(layer, torch.nn.Conv2d):
                layer_para['conv2d'] = param_conv2d(layer)
            elif isinstance(layer, torch.nn.ConvTranspose2d):
                trans_count += 1
                layer_para[f'trans2d_{trans_count}'] = param_trans2d(layer)
            elif isinstance(layer, torch.nn.Linear):
                layer_para['linear'] = param_linear(layer)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                layer_para['batchnorm'] = param_batchnorm(layer)
            elif isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.LeakyReLU):
                layer_para['relu'] = {}
            elif isinstance(layer, torch.nn.Tanh):
                layer_para['tanh'] = {}
            else:
                raise ValueError(f"Layer {layer} is not supported")
        return layer_para
    