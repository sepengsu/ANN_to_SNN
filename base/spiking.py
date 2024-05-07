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
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
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
    

class first_Spiking(nn.Module):
    def __init__(self, block, T):
        super(first_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.is_first = True
        self.idem = False
        self.sign = True  # 여기서는 sign을 사용하지 않습니다.

    def forward(self, x):
        if self.idem:
            return x

        # Prepare charges by expanding the input tensor across the time dimension
        x = x.unsqueeze(1)  # Add a time dimension
        x = x.repeat(1, self.T, 1)  # Repeat input across the time dimension

        # Flatten the batch and time dimensions for processing through the block
        train_shape = [x.shape[0], x.shape[1]]  # Original dimensions of batch and time
        x = x.flatten(0, 1)  # Flatten batch and time dimensions
        x = self.block(x)  # Process through the provided block
        train_shape.extend(x.shape[1:])  # Append the remaining dimensions after processing
        x = x.reshape(train_shape)  # Reshape to the original batch and new dimensions

        # Here, no spiking or threshold logic is applied, just linear transformation
        return x

    def extra_repr(self) -> str:
        return f'T={self.T}, is_first={self.is_first}, idem={self.idem}'
