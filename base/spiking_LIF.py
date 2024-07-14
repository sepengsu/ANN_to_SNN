import torch
import torch.nn as nn

import torch.nn as nn

class Spiking_LIF(nn.Module):
    def __init__(self, block, T, tau=20.0):
        super(Spiking_LIF, self).__init__()
        self.block = block
        self.T = T
        self.tau = tau  # Leaky 통합을 위한 시간 상수
        self.is_first = False
        self.idem = False
        self.sign = True
        
    def forward(self, x):
        if self.idem:
            return x
        
        # 막전위를 임계값의 절반으로 초기화
        # 이 부분에서 올바른 인덱스로 접근하도록 수정
        threshold = None
        for layer in self.block:
            if hasattr(layer, 'act_alpha'):
                threshold = layer.act_alpha.data
                break
        if threshold is None:
            raise ValueError("act_alpha가 있는 레이어를 찾을 수 없습니다.")
        
        membrane = 0.5 * threshold
        sum_spikes = 0
        
        # charges 준비
        if self.is_first:
            x.unsqueeze_(1)
            x = x.repeat(1, self.T, 1, 1, 1)
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        # charges 통합 (누수 포함)
        for dt in range(self.T):
            membrane = (1 - 1/self.tau) * membrane + x[:, dt] # 누수 포함
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:], device=membrane.device)
            
            spikes = membrane >= threshold # threshold를 넘으면 spike 발생
            membrane[spikes] = membrane[spikes] - threshold # spike 발생한 부분은 임계값만큼 감소
            spikes = spikes.float() # spike 발생한 부분은 1로 표시
            sum_spikes = sum_spikes + spikes
            
            # 서명된 스파이크
            if self.sign: # 서명된 스파이크 사용
                inhibit = membrane <= -1e-3 # 임계값의 음수 부분을 inhibit으로 설정
                inhibit = inhibit & (sum_spikes > 0) # inhibit이 발생하고 spike가 있을 때
                membrane[inhibit] = membrane[inhibit] + threshold # inhibit이 발생한 부분은 임계값만큼 증가
                inhibit = inhibit.float() # inhibit이 발생한 부분은 1로 표시
                sum_spikes = sum_spikes - inhibit # inhibit이 발생한 부분은 spike를 제거
            else:
                inhibit = 0

            spike_train[:, dt] = spikes - inhibit # spike와 inhibit을 합침
        
        spike_train = spike_train * threshold
        return spike_train



class last_Spiking_LIF(nn.Module):
    def __init__(self, block, T, tau=20.0):
        super(last_Spiking_LIF, self).__init__()
        self.block = block
        self.T = T
        self.tau = tau  # Leaky 통합을 위한 시간 상수
        self.is_first = False
        self.idem = False
        self.sign = True

    def forward(self, x):
        if self.idem:
            return x
        
        # 막전위를 임계값의 절반으로 초기화
        threshold = None
        for layer in self.block:
            if hasattr(layer, 'act_alpha'):
                threshold = layer.act_alpha.data
                break
        if threshold is None:
            raise ValueError("act_alpha가 있는 레이어를 찾을 수 없습니다.")
        
        membrane = 0.5 * threshold
        sum_spikes = 0

        # charges 준비
        if self.is_first:
            x.unsqueeze_(1)
            x = x.repeat(1, self.T, 1, 1, 1)
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        # charges 통합 (누수 포함)
        for dt in range(self.T):
            membrane = (1 - 1/self.tau) * membrane + x[:, dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],\
                                           device=membrane.device)
            
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            # 서명된 스파이크
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:, dt] = spikes - inhibit
        
        spike_train = spike_train * threshold
        return spike_train.sum(dim=1)