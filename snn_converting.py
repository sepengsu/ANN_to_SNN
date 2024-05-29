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
from model.vae import Quant_VAE, S_VAE
from model.vae_v2 import LIF_VAE_v2
from origin.ann_vae import VanillaVAE
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


def rename_keys_for_compatibility(model_state_dict):
    key_map = {
        'fc_mu.block.weight': 'fc_mu.block.0.weight',
        'fc_mu.block.bias': 'fc_mu.block.0.bias',
        'fc_var.block.weight': 'fc_var.block.0.weight',
        'fc_var.block.bias': 'fc_var.block.0.bias',
        'decoder_input.block.weight': 'decoder_input.block.0.weight',
        'decoder_input.block.bias': 'decoder_input.block.0.bias',
    }
    
    new_state_dict = {}
    for old_key, value in model_state_dict.items():
        new_key = key_map.get(old_key, old_key)  # 변환 규칙에 따라 새 키를 찾거나 기존 키를 사용
        new_state_dict[new_key] = value
    return new_state_dict

def test(network, testloader):
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

            print(f'Test [{batch_idx}/{len(testloader)}] Loss: {loss_meter.avg}, RECONS: {recons_meter.avg}, KLD: {kld_meter.avg}')

            if batch_idx == len(testloader)-1:
                os.makedirs(f'checkpoint/{args.after_name}/imgs/test/', exist_ok=True)
                torchvision.utils.save_image((real_img+1)/2, f'checkpoint/{args.after_name}/imgs/test/convert_input.png')
                torchvision.utils.save_image((recons+1)/2, f'checkpoint/{args.after_name}/imgs/test/convert_recons.png')
                writer.add_images('Test/input_img', (real_img+1)/2,1)
                writer.add_images('Test/recons_img', (recons+1)/2,1)

    logging.info(f"Test Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} KLD: {kld_meter.avg}")
    writer.add_scalar('Test/loss', loss_meter.avg)
    writer.add_scalar('Test/recons_loss', recons_meter.avg)
    writer.add_scalar('Test/kld', kld_meter.avg)
    return loss_meter.avg

def sample(network, batch_size=128):
    network = network.eval()
    with torch.no_grad():
        samples = network.sample(batch_size, device)
        writer.add_images('Sample/sample_img', (samples+1)/2)
        os.makedirs(f'checkpoint/{args.after_name}/imgs/sample/', exist_ok=True)
        torchvision.utils.save_image((samples+1)/2, f'checkpoint/{args.after_name}/imgs/sample/convert_sample.png')

def calc_inception_score(network,batch_size=256):
    network = network.eval()
    with torch.no_grad():
        inception_mean, inception_std = inception_score.get_inception_score_ann(network, device=device, batch_size=batch_size, batch_times=10)
        writer.add_scalar('Sample/inception_score_mean', inception_mean,1)
        writer.add_scalar('Sample/inception_score_std', inception_std, 1)

def calc_clean_fid(network):
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
        writer.add_scalar('Sample/FID', fid_score, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelname', type=str, default='s_vae',help='model name (s_vae, lif_vae, lif_vae_v2, vanilla_vae)')
    parser.add_argument('-before_name', type=str,required=True)
    parser.add_argument('-after_name', type=str,required=True)
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-batch_size', type=int, default=250)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-device', type=int, default=torch.cuda.current_device() if torch.cuda.is_available() else None)
    # quantization arguments
    parser.add_argument('-bit', type=int, default=8)
    parser.add_argument('--quant', action='store_true', default=False)
    parser.add_argument('--unsigned', action='store_true', default=False)

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
    modelname = args.modelname.upper() if args.modelname.lower() in ['s_vae', 'lif_vae', 'lif_vae_v2', 'vanilla_vae'] else 'LIF_VAE_v2'
    if args.dataset.lower() == 'mnist':     
        train_loader, test_loader = load_dataset_ann.load_mnist(data_path, args.batch_size)
        in_channels = 1 
        strings = f'net = {modelname}({in_channels}, {args.latent_dim},T = 2**{args.bit} - 1)'
        exec(strings)
        # net = Quant_VAE(in_channels, args.latent_dim)
    elif args.dataset.lower() == 'fashion':
        train_loader, test_loader = load_dataset_ann.load_fashionmnist(data_path, args.batch_size)
        in_channels = 1
        net = S_VAE(in_channels, args.latent_dim,T = 2**args.bit - 1)
    elif args.dataset.lower() == 'celeba':
        train_loader, test_loader = load_dataset_ann.load_celeba(data_path, args.batch_size)
        in_channels = 3
        net = S_VAE(in_channels, args.latent_dim,T = 2**args.bit - 1)
        raise ValueError("celeba dataset is not supported")
        # net = ann_vae.VanillaVAELarge(in_channels, args.latent_dim)
    elif args.dataset.lower() == 'cifar10':
        train_loader, test_loader = load_dataset_ann.load_cifar10(data_path, args.batch_size)
        in_channels = 3
        net = S_VAE(in_channels, args.latent_dim,T = 2**args.bit - 1)
    else:
        raise ValueError("invalid dataset")
    

    
    # quantinize
    if args.quant:
        net = quantinize(net, args)
    if args.unsigned:
        unsigned_spikes(net)
    net = net.to(device)

    writer = SummaryWriter(log_dir=f'checkpoint/{args.after_name}/tb')
    logging.basicConfig(filename=f'checkpoint/{args.after_name}/{args.after_name}.log', level=logging.INFO)
    
    logging.info(args)

    # load checkpoint
    args.checkpoint = f'checkpoint/{args.before_name}/best.pth'
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        checkpoint = rename_keys_for_compatibility(checkpoint) # 키 이름 변경
        net.load_state_dict(checkpoint)  

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    best_loss = 1e8
    e  = 1
    test_loss = test(net, test_loader)

    sample(net,128)
    calc_inception_score(net)
    # calc_clean_fid(net)

    writer.close()