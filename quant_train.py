import os
import os.path
import numpy as np
import logging
import argparse
# import pycuda.driver as cuda

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
from utils.util_file import AverageMeter

from utils.datasets import load_dataset_ann
import metrics.inception_score as inception_score
import metrics.clean_fid as clean_fid
# import metrics.autoencoder_fid as autoencoder_fid # 일단 주석처리
from base.quant_layer import QuantConv2d,QuantizedFC, QuantTrans2d, QuantLinear
from base.quant_layer import QuantReLU
from base.quant_dif import QuantTanh, QuantLeakyReLU
from base.quant_layer import build_power_value, weight_quantize_fn, act_quantization
# from model.vae import Quant_VAE
from model.vae_final import Quant_VAE
from base.spiking import unsigned_spikes


max_accuracy = 0
min_loss = 1000



def quantinize(model, args):
    for m in model.modules():
        #Ouroboros-------determine quantization
        #APoT quantization for weights, uniform quantization for activations
        if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantTrans2d):
            #weight quantization, use APoT
            m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
        if isinstance(m, QuantReLU) or isinstance(m, QuantTanh):
            #activation quantization, use uniform
            m.act_grid = build_power_value(args.bit)
            m.act_alq = act_quantization(b=args.bit, grid=m.act_grid, power=False) 
    return model
   
def train(network, trainloader, opti, epoch):
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

        print(f'Train[{epoch}/{max_epoch}] [{batch_idx}/{len(trainloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, KLD: {kld_meter.avg}')

        if batch_idx == len(trainloader)-1:
            os.makedirs(f'checkpoint/{args.name}/imgs/train/', exist_ok=True)
            torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_input.png')
            torchvision.utils.save_image((recons+1)/2, f'checkpoint/{args.name}/imgs/train/epoch{epoch}_recons.png')
            writer.add_images('Train/input_img', (real_img+1)/2, epoch)
            writer.add_images('Train/recons_img', (recons+1)/2, epoch)

    logging.info(f"Train [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} KLD: {kld_meter.avg}")
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Train/kld', kld_meter.avg, epoch)

    return loss_meter.avg

def test(network, testloader, epoch):
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
                os.makedirs(f'checkpoint/{args.name}/imgs/test/', exist_ok=True)
                torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_input.png')
                torchvision.utils.save_image((recons+1)/2, f'checkpoint/{args.name}/imgs/test/epoch{epoch}_recons.png')
                writer.add_images('Test/input_img', (real_img+1)/2, epoch)
                writer.add_images('Test/recons_img', (recons+1)/2, epoch)

    logging.info(f"Test [{epoch}] Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} KLD: {kld_meter.avg}")
    writer.add_scalar('Test/loss', loss_meter.avg, epoch)
    writer.add_scalar('Test/recons_loss', recons_meter.avg, epoch)
    writer.add_scalar('Test/kld', kld_meter.avg, epoch)
    # writer.add_scalar('Test/mul', count_mul_add.mul_sum / len(testloader), epoch)
    # writer.add_scalar('Test/add', count_mul_add.add_sum / len(testloader), epoch)

    # for handle in hook_handles:
    #     handle.remove()

    return loss_meter.avg

def sample(network, epoch, batch_size=128):
    network = network.eval()
    with torch.no_grad():
        samples = network.sample(batch_size, device)
        writer.add_images('Sample/sample_img', (samples+1)/2, epoch)
        os.makedirs(f'checkpoint/{args.name}/imgs/sample/', exist_ok=True)
        torchvision.utils.save_image((samples+1)/2, f'checkpoint/{args.name}/imgs/sample/epoch{epoch}_sample.png')

def calc_inception_score(network, epoch, batch_size=256):
    network = network.eval()
    with torch.no_grad():
        inception_mean, inception_std = inception_score.get_inception_score_ann(network, device=device, batch_size=batch_size, batch_times=10)
        print(f'Inception Score: {inception_mean} +/- {inception_std}')
        writer.add_scalar('Sample/inception_score_mean', inception_mean, epoch)
        writer.add_scalar('Sample/inception_score_std', inception_std, epoch)

def calc_clean_fid(network, epoch):
    network = network.eval()
    with torch.no_grad():
        if args.dataset.lower() == 'mnist': 
            dataset_name = 'MNIST'
        elif args.dataset.lower() == 'fashion': 
            dataset_name = 'FashionMNIST'
        elif args.dataset.lower() == 'celeba': 
            dataset_name = 'celeba'
        elif args.dataset.lower() == 'cifar10': 
            dataset_name = 'cifar10'
        else:
            raise ValueError()

        fid_score = clean_fid.get_clean_fid_score_ann(network, dataset_name, device, 5000)
        writer.add_scalar('Sample/FID', fid_score, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-batch_size', type=int, default=250)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-device', type=int, default=torch.cuda.current_device() if torch.cuda.is_available() else None)
    parser.add_argument('--epoch', type=int, default=150)
    # quantization arguments
    parser.add_argument('-bit', type=int, default=8)
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

    if args.dataset.lower() == 'mnist':     
        train_loader, test_loader = load_dataset_ann.load_mnist(data_path, args.batch_size)
        in_channels = 1 
        net = Quant_VAE(in_channels, args.latent_dim)

    elif args.dataset.lower() == 'fashion':
        train_loader, test_loader = load_dataset_ann.load_fashionmnist(data_path, args.batch_size)
        in_channels = 1
        net = Quant_VAE(in_channels, args.latent_dim)
    elif args.dataset.lower() == 'celeba':
        train_loader, test_loader = load_dataset_ann.load_celeba(data_path, args.batch_size)
        in_channels = 3
        raise ValueError("celeba dataset is not supported")
        # net = ann_vae.VanillaVAELarge(in_channels, args.latent_dim)
    elif args.dataset.lower() == 'cifar10':
        train_loader, test_loader = load_dataset_ann.load_cifar10(data_path, args.batch_size)
        in_channels = 3
        net = Quant_VAE(in_channels, args.latent_dim)
    else:
        raise ValueError("invalid dataset")
    
    # quantinize
    if args.quant:
        net = quantinize(net, args)
    net = net.to(device)

    os.makedirs(f'checkpoint/{args.name}', exist_ok=True)

    writer = SummaryWriter(log_dir=f'checkpoint/{args.name}/tb')
    logging.basicConfig(filename=f'checkpoint/{args.name}/{args.name}.log', level=logging.INFO)
    
    logging.info(args)

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint)  

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, betas=(0.9, 0.999))
    best_loss = 1e8
    max_epoch = args.epoch
    for e in range(max_epoch):
        train_loss = train(net, train_loader, optimizer, e)
        test_loss = test(net, test_loader, e)
        torch.save(net.state_dict(), f'checkpoint/{args.name}/checkpoint.pth')
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(net.state_dict(), f'checkpoint/{args.name}/best.pth')

        sample(net, e, batch_size=128)

        calc_inception_score(net, e)
        # calc_clean_fid(net, e)
        
    print("Training is finished")    
    writer.close()