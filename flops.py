import torch
import torch.nn as nn

class FLOPsCalculator:
    def __init__(self, model, input_size):
        """
        모델과 입력 크기를 받아서 FLOPs와 파라미터 수를 계산하는 클래스
        :param model: PyTorch 모델
        :param input_size: 모델에 대한 입력 크기 (배치 크기, 채널, 높이, 너비)
        """
        self.model = model
        self.input_size = input_size
        self.total_flops = 0
        self.total_params = 0
        self.device = next(model.parameters()).device  # 모델이 위치한 장치 확인
        self.calculate_flops_params()

    def calculate_flops_params(self):
        """
        모델의 모든 레이어를 순회하면서 FLOPs와 파라미터 수를 계산
        """
        # 입력 텐서를 모델과 동일한 장치로 이동
        input_tensor = torch.randn(self.input_size).to(self.device)

        # Hook을 사용하여 각 레이어에 대해 연산량과 파라미터 수를 계산
        def hook_fn(layer, inputs, outputs):
            flops, params = 0, 0
            # Conv2D FLOPs 계산
            if isinstance(layer, nn.Conv2d):
                flops = self.conv2d_flops(layer, inputs[0].shape)
                params = sum(p.numel() for p in layer.parameters())
            # Linear (Fully Connected) FLOPs 계산
            elif isinstance(layer, nn.Linear):
                flops = self.fc_flops(layer)
                params = sum(p.numel() for p in layer.parameters())
            # BatchNorm, Activation 등 기타 연산은 무시할 수 있음
            elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU)):
                params = sum(p.numel() for p in layer.parameters())

            self.total_flops += flops
            self.total_params += params

        # Hook을 각 레이어에 등록
        hooks = []
        for layer in self.model.modules():
            hooks.append(layer.register_forward_hook(hook_fn))

        # 입력 데이터를 한 번 통과시켜 FLOPs를 계산
        self.model(input_tensor)

        # Hook 제거
        for hook in hooks:
            hook.remove()
            
        return self.total_flops, self.total_params

    def conv2d_flops(self, layer, input_size):
        """ Conv2D 레이어의 FLOPs 계산 """
        batch_size, in_channels, in_height, in_width = input_size
        out_channels = layer.out_channels
        kernel_height, kernel_width = layer.kernel_size
        out_height = (in_height - kernel_height + 2 * layer.padding[0]) // layer.stride[0] + 1
        out_width = (in_width - kernel_width + 2 * layer.padding[1]) // layer.stride[1] + 1

        flops = 2 * in_channels * out_channels * kernel_height * kernel_width * out_height * out_width
        return flops

    def fc_flops(self, layer):
        """ Fully Connected 레이어의 FLOPs 계산 """
        in_features = layer.in_features
        out_features = layer.out_features
        flops = 2 * in_features * out_features
        return flops

class SLOPsCalculator:
    def __init__(self, model, input_size, T=3):
        """
        모델과 입력 크기 및 time steps (T)을 받아서 SLOPs와 파라미터 수를 계산하는 클래스
        :param model: PyTorch 모델
        :param input_size: 모델에 대한 입력 크기 (배치 크기, 채널, 높이, 너비)
        :param T: Spiking 뉴런 네트워크에서 시간 스텝
        """
        self.model = model
        self.input_size = input_size
        self.T = T
        self.total_slops = 0
        self.total_params = 0
        self.device = next(model.parameters()).device  # 모델이 위치한 장치 확인
        self.calculate_slops_params()

    def calculate_slops_params(self):
        """
        모델의 모든 레이어를 순회하면서 SLOPs와 파라미터 수를 계산
        """
        # 입력 텐서를 모델과 동일한 장치로 이동
        input_tensor = torch.randn(self.input_size).to(self.device)

        # Hook을 사용하여 각 레이어에 대해 연산량과 파라미터 수를 계산
        def hook_fn(layer, inputs, outputs):
            slops, params = 0, 0
            # Conv2D SLOPs 계산
            if isinstance(layer, nn.Conv2d):
                slops = self.spiking_layer_slops(layer, inputs[0].shape)
                params = sum(p.numel() for p in layer.parameters())
            # Spiking 레이어 SLOPs 계산
            elif hasattr(layer, 'spiking'):
                slops = self.spiking_layer_slops(layer, inputs[0].shape)
                params = sum(p.numel() for p in layer.parameters())

            self.total_slops += slops
            self.total_params += params

        # Hook을 각 레이어에 등록
        hooks = []
        for layer in self.model.modules():
            hooks.append(layer.register_forward_hook(hook_fn))

        # 입력 데이터를 한 번 통과시켜 SLOPs를 계산
        self.model(input_tensor)

        # Hook 제거
        for hook in hooks:
            hook.remove()

        return self.total_slops, self.total_params

    def spiking_layer_slops(self, layer, input_size):
        """Spiking Conv2D 레이어의 SLOPs 계산 (시간 스텝 T 포함)"""
        batch_size, in_channels, in_height, in_width = input_size
        out_channels = layer.out_channels
        kernel_height, kernel_width = layer.kernel_size
        out_height = (in_height - kernel_height + 2 * layer.padding[0]) // layer.stride[0] + 1
        out_width = (in_width - kernel_width + 2 * layer.padding[1]) // layer.stride[1] + 1

        # Spiking Conv2D는 시간 스텝(T)을 곱하여 SLOPs 계산
        slops = 2 * in_channels * out_channels * kernel_height * kernel_width * out_height * out_width
        return slops * self.T
    
if __name__ == '__main__':
    from model.vae_IF import IF_VAE, Quant_VAE
    model1 = Quant_VAE(3, 128)
    model2 = IF_VAE(3, 128)
    input_size = (1, 3, 32, 32)
    flops_analyzer = FLOPsCalculator(model1, input_size)
    total_flops, total_params = flops_analyzer.calculate_flops_params()
    print(f"Quant_VAE: Total FLOPs: {total_flops}, Total Params: {total_params}")
    slops_analyzer = SLOPsCalculator(model2, input_size)
    total_slops, total_params = slops_analyzer.calculate_slops_params()
    print(f"IF_VAE: Total SLOPs: {total_slops}, Total Params: {total_params}")