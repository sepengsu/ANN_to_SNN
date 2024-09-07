import os
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
import psutil
import logging
import time
from fvcore.nn import FlopCountAnalysis

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

class Monitor:
    def __init__(self, name, log_dir='log', model_dir='checkpoint'):
        times = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
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
            from zeus.monitor import ZeusMonitor
            import pynvml
            self.use_zeus = True
            self.monitor = ZeusMonitor()
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            logger.warning("CUDA is not available")
            self.use_zeus = False

    def start_monitoring(self, pids):
        if self.use_zeus and self.use_cuda:
            self.monitor.measure(pids=pids, devices=[torch.device("cuda")])
        else:
            self.cpu_energy_usage.append(psutil.cpu_percent(interval=None))

    def stop_monitoring(self):
        if self.use_zeus and self.use_cuda:
            results = self.monitor.get_results()
            for device, metrics in results.items():
                for metric, value in metrics.items():
                    self.writer.add_scalar(f"Zeus/{device}/{metric}", value, 0)
            import pynvml
            pynvml.nvmlShutdown()
        else:
            avg_cpu_energy = sum(self.cpu_energy_usage) / len(self.cpu_energy_usage)
            self.writer.add_scalar('CPU_Energy/Usage', avg_cpu_energy, 0)
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
                        self.writer.add_scalar(f"Profiling/{event.key}/cuda_memory_usage", event.cuda_memory_usage, epoch)
                    if hasattr(event, 'self_cuda_memory_usage'):
                        self.writer.add_scalar(f"Profiling/{event.key}/self_cuda_memory_usage",
                                                event.self_cuda_memory_usage,
                                                epoch)

                if hasattr(event, 'flops'):
                    self.writer.add_scalar(f"Profiling/{event.key}/flops", event.flops / 1e6, epoch)

        except AssertionError as e:
            logger.error(f"Assertion error during profiling: {e}")
            logger.debug("AssertionError 발생 시 프로파일러 상태:")
            for key, value in vars(prof).items():
                logger.debug(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Unexpected error during profiling: {e}")

    def start_profiling(self, max_epoch, network, train_fn, test_fn, train_loader, test_loader, optimizer):
        best_loss = 1e8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=100, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for epoch in range(max_epoch):
                self.start_monitoring([os.getpid()])

                logger.info(f"Profiler action before start: {prof.current_action}")

                # Calculate FLOPs for the model
                input_tensor = next(iter(train_loader))[0].to(device)
                flops_analyzer = FlopCountAnalysis(network, input_tensor)
                total_flops = flops_analyzer.total()
                self.writer.add_scalar('FLOPs/Total', total_flops, epoch)

                train_loss = train_fn(network, train_loader, optimizer, epoch, self)
                test_loss = test_fn(network, test_loader, epoch, self)

                prof.step()

                self.log_profiling_info(prof, epoch)
                self.stop_monitoring()

                torch.save(network.state_dict(), f'{self.checkpoint_dir}/checkpoint.pth')
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(network.state_dict(), f'{self.checkpoint_dir}/best.pth')

if __name__ == "__main__":
    import torch
    print(torch.__version__)
