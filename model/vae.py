import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from base.quant_layer import QuantConv2d,QuantTrans2d,QuantLinear,QuantReLU, QuantTanh, first_conv, last_trans2d
from base.spiking import Spiking, last_Spiking, IF
from origin.ann_vae import VanillaVAE
from converting.utils import Params

class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)

class Quant_VAE(VanillaVAE):
    def __init__(self, in_channels, latent_dim) -> None:
        super().__init__(in_channels, latent_dim)

        # encoder 재정의 
        encoder = []   
        for i in range(len(self.encoder)):
            seq = self.encoder[i]
            params = Params(seq).get_params()
            if i==0: # 첫번째 layer만 first_layer 적용
                encoder.append(Dummy(nn.Sequential(
                    first_conv(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    QuantReLU(),
                )))
            else:
                encoder.append(Dummy(nn.Sequential(
                    QuantConv2d(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    QuantReLU(),
                )))
        self.encoder = nn.Sequential(*encoder)

        # fc_mu, fc_var 재정의
        params = Params(self.fc_mu).get_params()
        self.fc_mu = Dummy(nn.Sequential(
            QuantLinear(**params['linear']),
        ))
        params = Params(self.fc_var).get_params()
        self.fc_var = Dummy(nn.Sequential(
            QuantLinear(**params['linear']),
        ))

        # decoder 재정의
        decoder = []
        for i in range(len(self.decoder)):
            seq = self.decoder[i]
            params = Params(seq).get_params()
            decoder.append(Dummy(nn.Sequential(
                QuantTrans2d(**params['trans2d']),
                nn.BatchNorm2d(**params['batchnorm']),
                QuantReLU(inplace=True),
            )))

        self.decoder = nn.Sequential(*decoder)
        self.decoder_input = Dummy(nn.Sequential(
            QuantLinear(in_features=latent_dim, out_features=self.hidden_dims[-1] * 4),
        ))
        # final_layer 재정의
        params = Params(self.final_layer).get_last_params()
        self.final_layer = Dummy(nn.Sequential(
            QuantTrans2d(**params['trans2d_1']),
            nn.BatchNorm2d(**params['batchnorm']),
            QuantReLU(inplace=True),
            last_trans2d(**params['trans2d_2']), # 마지막은 8bit로
            QuantTanh(inplace=True)
        ))





class S_VAE(VanillaVAE):
    def __init__(self, in_channels, latent_dim,T: int = 3) -> None:
        super().__init__(in_channels, latent_dim)

        # encoder 재정의
        for i in range(len(self.encoder)):
            seq = self.encoder[i]
            params = Params(seq).get_params()
            if i==0:
                self.encoder[i] = Spiking(nn.Sequential(
                    first_conv(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    IF()), T)
            else:
                self.encoder[i] = Spiking(nn.Sequential(
                    QuantConv2d(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    IF()), T)
            
        # fc_mu, fc_var 재정의
        params = Params(self.fc_mu).get_params()
        self.fc_mu = Spiking(nn.Sequential(
            QuantLinear(**params['linear']),
            IF()), T)
        params = Params(self.fc_var).get_params()
        self.fc_var = Spiking(nn.Sequential(
            QuantLinear(**params['linear']),
            IF()), T)
        
        # decoder 재정의
        for i in range(len(self.decoder)):
            seq = self.decoder[i]
            params = Params(seq).get_params()
            self.decoder[i] = Spiking(nn.Sequential(
                QuantTrans2d(**params['trans2d']),
                nn.BatchNorm2d(**params['batchnorm']),
                IF()), T)
        
        # final_layer 재정의
        params = Params(self.final_layer).get_last_params()
        self.final_layer = Spiking(nn.Sequential(
            QuantTrans2d(**params['trans2d_1']),
            nn.BatchNorm2d(**params['batchnorm']),
            IF(),
            last_trans2d(**params['trans2d_2']),
            IF()), T)
        
