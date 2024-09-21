import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.util_file import AverageMeter
from utils.datasets import load_dataset_ann
from monitoring import Monitor
import logging
from helper import quantinize, test, sample, calc_inception_score, calc_clean_fid
from model import vae_IF, vae_LIF

max_accuracy = 0
min_loss = 1000

def train(network, trainloader, opti, epoch, monitor):
    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    kld_meter = AverageMeter()
    
    network = network.train()

    for batch_idx, (real_img, label) in enumerate(trainloader):         
        opti.zero_grad()
        real_img = real_img.to(device)
        recons, mu, log_var = network(real_img)
        losses = network.loss_function(recons, real_img, mu, log_var, 1/len(trainloader))
        losses['loss'].backward()

        opti.step()

        loss_meter.update(losses['loss'].detach().cpu().item())
        recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
        kld_meter.update(losses['KLD'].detach().cpu().item())

        print(f'Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg: .4f} , RECONS: {recons_meter.avg: .4f}, KLD: {kld_meter.avg: .4f}')

        if batch_idx == len(trainloader)-1:
            os.makedirs(f'{monitor.checkpoint_dir}/imgs/train/', exist_ok=True)

            # Convert tensors to CPU and scale to [0, 1] range
            real_img_cpu = (real_img.cpu() + 1) / 2
            recons_cpu = (recons.cpu() + 1) / 2

            # Save images
            torchvision.utils.save_image(real_img_cpu, f'{monitor.checkpoint_dir}/imgs/train/epoch{epoch}_input.png')
            torchvision.utils.save_image(recons_cpu, f'{monitor.checkpoint_dir}/imgs/train/epoch{epoch}_recons.png')

            monitor.writer.add_images('Train/input_img', real_img_cpu, epoch)
            monitor.writer.add_images('Train/recons_img', recons_cpu, epoch)

    logging.info(f"Train [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} KLD: {kld_meter.avg}")
    monitor.writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    monitor.writer.add_scalar('Train/recons_loss', recons_meter.avg, epoch)
    monitor.writer.add_scalar('Train/kld', kld_meter.avg, epoch)

    return loss_meter.avg


def test(network, testloader, epoch, monitor):
    loss_meter = AverageMeter()
    recons_meter = AverageMeter()
    kld_meter = AverageMeter()

    network = network.eval()
    with torch.no_grad():
        for batch_idx, (real_img, label) in enumerate(testloader):         
            real_img = real_img.to(device)
            recons, mu, log_var = network(real_img)
            losses = network.loss_function(recons, real_img, mu, log_var, 1/len(testloader))

            loss_meter.update(losses['loss'].detach().cpu().item())
            recons_meter.update(losses['Reconstruction_Loss'].detach().cpu().item())
            kld_meter.update(losses['KLD'].detach().cpu().item())

            print(f'Test[{epoch}/{max_epoch}] [{batch_idx}/{len(testloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, KLD: {kld_meter.avg}')

            if batch_idx == len(testloader)-1:
                print("Saving images", epoch, f"(data shape: {real_img.shape})")
                print("range: ", real_img.min(), real_img.max())
                os.makedirs(f'{monitor.checkpoint_dir}/imgs/test/', exist_ok=True)

                # Convert tensors to CPU and scale to [0, 1] range
                real_img_cpu = (real_img.cpu() + 1) / 2
                recons_cpu = (recons.cpu() + 1) / 2

                # Save images
                torchvision.utils.save_image(real_img_cpu, f'{monitor.checkpoint_dir}/imgs/test/epoch{epoch}_input.png')
                torchvision.utils.save_image(recons_cpu, f'{monitor.checkpoint_dir}/imgs/test/epoch{epoch}_recons.png')

                monitor.writer.add_images('Test/input_img', real_img_cpu, epoch)
                monitor.writer.add_images('Test/recons_img', recons_cpu, epoch)

    logging.info(f"Test [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} KLD: {kld_meter.avg}")
    monitor.writer.add_scalar('Test/loss', loss_meter.avg, epoch)
    monitor.writer.add_scalar('Test/recons_loss', recons_meter.avg, epoch)
    monitor.writer.add_scalar('Test/kld', kld_meter.avg, epoch)
    return loss_meter.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('-model', type=str, default='vae_IF', help='The name of model')
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-batch_size', type=int, default=250)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-device', type=int, default=torch.cuda.current_device() if torch.cuda.is_available() else None)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('-bit', type=int, default=8)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('--quant', action='store_true', help='quantize model')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    data_path = "./data"

    modeltype = args.model

    if args.dataset.lower() == 'mnist':     
        train_loader, test_loader = load_dataset_ann.load_mnist(data_path, args.batch_size)
        in_channels = 1 
        s = f"{modeltype}.Quant_VAE({in_channels}, {args.latent_dim})"
    elif args.dataset.lower() == 'miad':
        train_loader, test_loader = load_dataset_ann.load_MIAD_metal_welding(data_path, args.batch_size)
        in_channels = 3
        s = f"{modeltype}.Quant_VAE({in_channels}, {args.latent_dim})"
    else:
        raise ValueError("invalid dataset")
    net = eval(s)
    if args.quant:
        net = quantinize(net, args)
    net = net.to(device)

    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)
    print(f"Model: {s}, Dataset: {args.dataset}, Batch Size: {args.batch_size}, Latent Dim: {args.latent_dim}")
    print("Training is started")

    # 모니터 인스턴스 생성
    monitor = Monitor(args.name)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)  

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    max_epoch = args.epoch

    monitor.start_profiling(max_epoch, net, train, test, train_loader, test_loader, optimizer)

    monitor.stop_monitoring()

    print("Training is finished")
  