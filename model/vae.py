import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from base.quant_layer import QuantConv2d,QuantTrans2d,QuantLinear,QuantReLU, QuantTanh, first_conv, last_trans2d, QuantizedFC
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
        for i in range(len(self.encoder)):
            seq = self.encoder[i]
            params = Params(seq).get_params()
            if i==0: # 첫번째 layer만 first_layer 적용
                self.encoder[i] = Dummy(nn.Sequential(
                    first_conv(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    QuantReLU(),
                ))
            else:
                self.encoder[i]= Dummy(nn.Sequential(
                    QuantConv2d(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    QuantReLU(),
                ))

        # fc_mu, fc_var 재정의
        params = Params(self.fc_mu).get_params()
        self.fc_mu = Dummy(
            QuantizedFC(**params['linear']),
        )
        params = Params(self.fc_var).get_params()
        self.fc_var = Dummy(
            QuantizedFC(**params['linear']),
        )

        # decoder input 재정의
        params = Params(self.decoder_input).get_params()
        self.decoder_input = Dummy(
            QuantizedFC(**params['linear']),
        ) # decoder input은 8bit로 할지 고민해봐야함

        # decoder 재정의
        for i in range(len(self.decoder)):
            seq = self.decoder[i]
            params = Params(seq).get_params()
            self.decoder[i] = Dummy(nn.Sequential(
                QuantTrans2d(**params['trans2d']),
                nn.BatchNorm2d(**params['batchnorm']),
                QuantReLU(),
            ))
        # final_layer 재정의
        params = Params(self.final_layer).get_last_params()
        self.final_layer = Dummy(nn.Sequential(
            QuantTrans2d(**params['trans2d_1']),
            nn.BatchNorm2d(**params['batchnorm']),
            QuantReLU(inplace=True),
            last_trans2d(**params['trans2d_2']), # 마지막은 8bit로
            QuantTanh()
        ))
    
    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU)\
                         or isinstance(m, QuantTanh) or isinstance(m, QuantTrans2d) or isinstance(m, last_trans2d) or isinstance(m, QuantizedFC):
                m.show_params()

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
        self.fc_mu = last_Spiking(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        params = Params(self.fc_var).get_params()
        self.fc_var = last_Spiking(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        
        # decoder input 재정의
        params = Params(self.decoder_input).get_params()
        self.decoder_input = last_Spiking(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        
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
            last_trans2d(**params['trans2d_2']),
            QuantTanh(),
            IF()), T)
    
    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU)\
                         or isinstance(m, QuantTanh) or isinstance(m, QuantTrans2d) or isinstance(m, last_trans2d) or isinstance(m, QuantizedFC):
                m.show_params()
        
        
