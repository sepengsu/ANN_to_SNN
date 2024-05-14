import torch
import torch.nn as nn



def unsigned_spikes(model):
    for m in model.modules():
         if isinstance(m, Spiking):
             m.sign = False

#####the spiking wrapper######

class Spiking(nn.Module):
    def __init__(self, block, T):
        super(Spiking, self).__init__()
        self.block = block
        self.T = T
        self.is_first = False
        self.idem = False
        self.sign = True
    def forward(self, x):
        if self.idem:
            return x
        
        ###initialize membrane to half threshold
        threshold = self.block[2].act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0
        
        #prepare charges
        if self.is_first:
            x.unsqueeze_(1)
            x = x.repeat(1, self.T, 1, 1, 1)
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + \
                                          membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            ###signed spikes###
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:,dt] = spikes - inhibit
                
        spike_train = spike_train * threshold
        return spike_train


class last_Spiking(nn.Module):
    def __init__(self, block, T):
        super(last_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.idem = False
        
    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        return x.sum(dim=1)
    
class IF(nn.Module):
    def __init__(self):
        super(IF, self).__init__()
        ###changes threshold to act_alpha
        ###being fleet
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x

    def show_params(self):
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold activation alpha: {:2f}'.format(act_alpha)) 
    
    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)  
    
class Repeat(nn.Module):
    def __init__(self, block, T):
        super(Repeat, self).__init__()
        self.block = block  # 반복할 블록
        self.T = T  # 반복 횟수
        self.is_first = True  # 처음 실행 여부
        self.idem = False  # 멱등성 여부
        self.sign = True  # 여기서는 sign을 사용하지 않습니다.

    def forward(self, x):
        if self.idem:
            return x  # 멱등이 true인 경우 입력 그대로 반환

        # 시간 차원을 추가하여 입력 텐서를 확장
        x = x.unsqueeze(1)  # 시간 차원 추가
        x = x.repeat(1, self.T, 1)  # 시간 차원을 따라 입력 반복  ==> repeat 하는 것이 맞는가????

        # 블록 처리를 위해 배치 및 시간 차원을 펼침
        train_shape = [x.shape[0], x.shape[1]]  # 배치와 시간의 원래 차원
        x = x.flatten(0, 1)  # 배치와 시간 차원 펼치기
        x = self.block(x)  # 제공된 블록을 통해 처리
        train_shape.extend(x.shape[1:])  # 처리 후 남은 차원을 추가
        x = x.reshape(train_shape)  # 원래 배치와 새로운 차원으로 재구성

        # 여기서는 스파이킹 또는 임계값 로직이 적용되지 않고, 단순 선형 변환만 실행
        return x

    def extra_repr(self) -> str:
        return f'T={self.T}, is_first={self.is_first}, idem={self.idem}'



