import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from base.quant_layer import QuantConv2d,QuantTrans2d,QuantLinear,QuantReLU,first_conv, last_trans2d, QuantizedFC
from base.quant_dif import QuantTanh, QuantLeakyReLU
from base.spiking import last_Spiking, IF, Repeat
from base.spiking_v2 import Spiking_LIF
from origin.ann_vae import VanillaVAE
from converting.utils import Params

class LIF_VAE(VanillaVAE):
    '''
    최종본으로 사용하지는 않고 있음 주요 특징은 다음과 같다
    1. encoder, decoder, fc_mu, fc_var, decoder_input, final_layer를 spiking으로 변환
    2. encoder의 첫번째 레이어는 first_conv로 변환
    3. decoder_input은 Repeat으로 변환
    4. decoder는 Spiking_LIF로 변환
    5. final_layer는 last_Spiking으로 변환
    6. 각 레이어의 파라미터를 출력하는 show_params 메소드 추가
    7. Spiking_LIF의 sign을 False로 설정하여 signed spiking을 사용하지 않도록 설정
    8. Spiking_LIF의 idem을 False로 설정하여 항등함수를 사용하지 않도록 설정
    9. Repeat의 is_first를 True로 설정하여 첫번째 레이어임을 알림
    '''
    def __init__(self, in_channels, latent_dim,T: int = 3) -> None:
        super().__init__(in_channels, latent_dim)

        # encoder 재정의
        for i in range(len(self.encoder)):
            seq = self.encoder[i]
            params = Params(seq).get_params()
            if i==0:
                self.encoder[i] = Spiking_LIF(nn.Sequential(
                    first_conv(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    IF()), T)
            else:
                self.encoder[i] = Spiking_LIF(nn.Sequential(
                    QuantConv2d(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    IF()), T)
                
        self.encoder[0].is_first = True
            
        # fc_mu, fc_var 재정의
        params = Params(self.fc_mu).get_params()
        self.fc_mu = last_Spiking(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        params = Params(self.fc_var).get_params()
        self.fc_var = last_Spiking(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        
        # decoder input 재정의
        params = Params(self.decoder_input).get_params()
        self.decoder_input = Repeat(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        
        self.decoder_input.is_first = True
        
        decoder = []
        # decoder 재정의
        for i in range(len(self.decoder)):
            seq = self.decoder[i]
            params = Params(seq).get_params()
            decoder.append(Spiking_LIF(nn.Sequential(
                QuantTrans2d(**params['trans2d']),
                nn.BatchNorm2d(**params['batchnorm']),
                IF()), T))
            
        # final_layer 재정의
        params = Params(self.final_layer).get_last_params()
        decoder.append(Spiking_LIF(nn.Sequential(
            QuantConv2d(**params['trans2d_1']), 
            nn.BatchNorm2d(**params['batchnorm']),
            IF()), T))
        self.decoder = nn.Sequential(*decoder)

        self.final_layer = last_Spiking(nn.Sequential(
            last_trans2d(**params['trans2d_2']), # 마지막은 8bit로
            QuantTanh()), T)
        
        
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=2)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        # 여기서 result.shape = [B x T x D] 이므로 [B x T x D x 2 x 2]로 reshape
        result = result.view(-1,result.shape[1], self.hidden_dims[-1], 2, 2) # encoder에서 flatten한거 다시 reshape
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    
    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU)\
                         or isinstance(m, QuantTanh) or isinstance(m, QuantTrans2d):
                m.show_params()
        


class LIF_VAE_v2(VanillaVAE): # 최종 본으로 시영 되고 있음 
    '''
    최종본으로 사용하고 있음 주요 특징은 다음과 같다
    1. encoder, decoder, fc_mu, fc_var, decoder_input, final_layer를 spiking으로 변환
    2. encoder의 첫번째 레이어는 first_conv로 변환
    3. decoder_input은 Spiking_LIF로 변환
    4. decoder는 Spiking_LIF로 변환
    5. final_layer는 last_Spiking으로 변환 + IF_DE로 상용
    6. 각 레이어의 파라미터를 출력하는 show_params 메소드 추가
    7. Spiking_LIF의 sign을 False로 설정하여 signed spiking을 사용하지 않도록 설정
    8. Spiking_LIF의 idem을 False로 설정하여 항등함수를 사용하지 않도록 설정
    '''
    def __init__(self, in_channels, latent_dim, T: int = 3) -> None:
        super().__init__(in_channels, latent_dim)

        # encoder 재정의
        for i in range(len(self.encoder)):
            seq = self.encoder[i]
            params = Params(seq).get_params()
            if i == 0:
                self.encoder[i] = Spiking_LIF(nn.Sequential(
                    first_conv(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    IF()), T)
            else:
                self.encoder[i] = Spiking_LIF(nn.Sequential(
                    QuantConv2d(**params['conv2d']),
                    nn.BatchNorm2d(**params['batchnorm']),
                    IF()), T)
                
        self.encoder[0].is_first = True
            
        # fc_mu, fc_var 재정의
        params = Params(self.fc_mu).get_params()
        self.fc_mu = last_Spiking(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        params = Params(self.fc_var).get_params()
        self.fc_var = last_Spiking(nn.Sequential(
            QuantizedFC(**params['linear'])), T)
        
        # decoder input 재정의
        params = Params(self.decoder_input).get_params()
        self.decoder_input = Spiking_LIF(nn.Sequential(
            QuantizedFC(**params['linear']),IF()), T) # IF 추가 및 스파킹 method 추가 
        self.decoder_input.is_first = True
        
        decoder = []
        # decoder 재정의
        for i in range(len(self.decoder)):
            seq = self.decoder[i]
            params = Params(seq).get_params()
            decoder.append(Spiking_LIF(nn.Sequential(
                QuantTrans2d(**params['trans2d']),
                nn.BatchNorm2d(**params['batchnorm']),
                IF()), T))
            
        params = Params(self.final_layer).get_last_params()
        decoder.append(Spiking_LIF(nn.Sequential(
                QuantTrans2d(**params['trans2d_1']),
                nn.BatchNorm2d(**params['batchnorm']),
                IF()), T))
        
        self.decoder = nn.Sequential(*decoder)

        self.final_layer = last_Spiking(nn.Sequential(
            last_trans2d(**params['trans2d_2']), # 마지막은 8bit로
            QuantTanh()), T)
        
        
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=2)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        # 여기서 result.shape = [B x T x D] 이므로 [B x T x D x 2 x 2]로 reshape
        result = result.view(-1, result.shape[1], self.hidden_dims[-1], 2, 2) # encoder에서 flatten한거 다시 reshape
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    
    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU) \
                    or isinstance(m, QuantTanh) or isinstance(m, QuantTrans2d):
                m.show_params()
