import os, re
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import psutil
import logging
import time
from flops import FLOPsCalculator  # 이미 존재하는 FLOPsCalculator를 사용
from flops import SLOPsCalculator  # 새롭게 추가할 SLOPsCalculator
import pynvml
from model.vae_IF import Quant_VAE, IF_VAE

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
    return logger

logger = setup_logger(__name__)

def numbering(name, log_dir):
    dir_list = os.listdir(f"./{log_dir}")
    dir_list = [filename for filename in dir_list if name in filename]
    if not any([name in filename for filename in dir_list]):
        return 1
    numbers = [int(filename.split('_')[-1]) for filename in dir_list]
    max_number = max(numbers) if numbers else 0
    return max_number + 1

class Monitor:
    def __init__(self, name, log_dir='log', model_dir='checkpoint'):
        times = numbering(name, log_dir)
        self.log_dir = f'{log_dir}/{name}_{times}'
        self.checkpoint_dir = f'{model_dir}/{name}_{times}'
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.name = name
        self.cpu_energy_usage = []
        self.use_cuda = torch.cuda.is_available()

        logger.info(f"Monitor initialized at: {self.log_dir}")
        logger.info(f"for tensorboard using this command: tensorboard --logdir={self.log_dir}")

        if self.use_cuda:
            logger.info("CUDA is available")
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            logger.warning("CUDA is not available")

    def start_monitoring(self, pids):
        if self.use_cuda:
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError as e:
                logger.error(f"NVML initialization error: {e}")
                return

            # GPU monitoring using PyNVML
            gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # mW to W
            gpu_mem_used = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024 ** 2)  # bytes to MB
            logger.info(f"GPU Power Usage: {gpu_power} W, GPU Memory Usage: {gpu_mem_used} MB")
            self.writer.add_scalar('energy/GPU/Power_Usage_W', gpu_power, 0)
            self.writer.add_scalar('energy/GPU/Memory_Usage_MB', gpu_mem_used, 0)

        # 항상 CPU 사용량 측정
        cpu_usage = psutil.cpu_percent(interval=None)  # CPU 사용량 측정
        self.cpu_energy_usage.append(cpu_usage)
        logger.info(f"CPU Usage: {cpu_usage}%")
        self.writer.add_scalar('energy/CPU/Usage_Percentage', cpu_usage, 0)

    def stop_monitoring(self):
        # GPU 모니터링 종료
        if self.use_cuda:
            pynvml.nvmlShutdown()  # NVML 종료

        # CPU 사용량 평균 계산
        avg_cpu_energy = sum(self.cpu_energy_usage) / len(self.cpu_energy_usage) if self.cpu_energy_usage else 0
        logger.info(f"Average CPU Energy Usage: {avg_cpu_energy}%")
        self.writer.add_scalar('energy/CPU_Energy/Avg_Usage_Percentage', avg_cpu_energy, 0)

        # TensorBoard writer 닫기
        self.writer.close()

    def log_profiling_info(self, prof, epoch):
        if prof is None:
            logger.warning("Profiler has not been initialized.")
            return
        try:
            key_averages = prof.key_averages()

            if not key_averages:
                logger.error("No profiling data available.")
                return

            for event in key_averages:
                if self.use_cuda:
                    if hasattr(event, 'cuda_memory_usage'):
                        self.writer.add_scalar(f"energy/Profiling/{event.key}/cuda_memory_usage", event.cuda_memory_usage, epoch)
                    if hasattr(event, 'self_cuda_memory_usage'):
                        self.writer.add_scalar(f"energy/Profiling/{event.key}/self_cuda_memory_usage", event.self_cuda_memory_usage, epoch)

        except AssertionError as e:
            logger.error(f"Assertion error during profiling: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during profiling: {e}")

    def start_profiling(self, max_epoch, network, train_fn, test_fn, train_loader, test_loader, optimizer):
        best_loss = 1e8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 모델을 GPU 또는 CPU로 이동 (사용 가능 시)
        network.to(device)

        for epoch in range(max_epoch):
            # 에포크마다 프로파일러를 켜기
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),  # 필요한 구간만 프로파일링
                on_trace_ready=tensorboard_trace_handler(self.log_dir),  # TensorBoard 핸들러 추가
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                
                self.start_monitoring([os.getpid()])

                logger.info(f"Profiler action before start: {prof.current_action}")

                # 모델에 따라 FLOPs 또는 SLOPs 계산
                if isinstance(network, Quant_VAE):
                    flops_analyzer = FLOPsCalculator(network, next(iter(train_loader))[0].to(device).shape)
                    total_flops, total_params = flops_analyzer.calculate_flops_params()
                    self.writer.add_scalar('energy/FLOPs/Total', total_flops, epoch)
                    self.writer.add_scalar('energy/Params/Total', total_params, epoch)
                elif isinstance(network, IF_VAE):
                    slops_analyzer = SLOPsCalculator(network, next(iter(train_loader))[0].to(device).shape)
                    total_slops, total_params = slops_analyzer.calculate_slops_params()
                    self.writer.add_scalar('energy/SLOPs/Total', total_slops, epoch)
                    self.writer.add_scalar('energy/Params/Total', total_params, epoch)

                # 학습 및 테스트 실행
                train_loss = train_fn(network, train_loader, optimizer, epoch, self)
                test_loss = test_fn(network, test_loader, epoch, self)

                prof.step()  # 프로파일러 진행 단계

                self.log_profiling_info(prof, epoch)
                self.stop_monitoring()

                # 모델 상태 저장
                torch.save(network.state_dict(), f'{self.checkpoint_dir}/checkpoint.pth')
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(network.state_dict(), f'{self.checkpoint_dir}/best.pth')


if __name__ == "__main__":
    import torch
    print(torch.__version__)
