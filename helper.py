import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.util_file import AverageMeter
from utils.datasets import load_dataset_ann
import metrics.inception_score as inception_score
import metrics.clean_fid as clean_fid
from base.quant_layer import QuantConv2d, QuantizedFC, QuantTrans2d, QuantLinear, QuantReLU
from base.quant_dif import QuantTanh, QuantLeakyReLU
from base.quant_layer import build_power_value, weight_quantize_fn, act_quantization
import logging
import torchprofile

def quantinize(model, args):
    for m in model.modules():
        if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantTrans2d):
            m.weight_quant = weight_quantize_fn(w_bit=args.bit, power=True)
        if isinstance(m, QuantReLU) or isinstance(m, QuantTanh):
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
        new_key = key_map.get(old_key, old_key)
        new_state_dict[new_key] = value
    return new_state_dict

def sample(network, epoch, batch_size, device, writer, dir_name):
    network = network.eval()
    with torch.no_grad():
        samples = network.sample(batch_size, device)
        writer.add_images('Sample/sample_img', (samples+1)/2, epoch)
        os.makedirs(f'{dir_name}/imgs/sample/', exist_ok=True)
        torchvision.utils.save_image((samples+1)/2, f'{dir_name}/imgs/sample/epoch{epoch}_sample.png')

def calc_inception_score(network, epoch, device, writer):
    network = network.eval()
    with torch.no_grad():
        inception_mean, inception_std = inception_score.get_inception_score_ann(network, device=device, batch_size=256, batch_times=10)
        writer.add_scalar('Sample/inception_score_mean', inception_mean, epoch)
        writer.add_scalar('Sample/inception_score_std', inception_std, epoch)

def calc_clean_fid(network, epoch, args, device, writer):
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
        elif args.dataset.lower() == 'miad': 
            dataset_name = 'MIAD_metal_welding'
        else:
            raise ValueError()

        fid_score = clean_fid.get_clean_fid_score_ann(network, dataset_name, device, 5000)
        writer.add_scalar('Sample/FID', fid_score, epoch)

def test(network, testloader, device, writer, dir_name, log_name):
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
                os.makedirs(f'{dir_name}/imgs/test/', exist_ok=True)
                torchvision.utils.save_image((real_img+1)/2, f'{dir_name}/imgs/test/convert_input.png')
                torchvision.utils.save_image((recons+1)/2, f'{dir_name}/imgs/test/convert_recons.png')
                writer.add_images('Test/input_img', (real_img+1)/2, 1)
                writer.add_images('Test/recons_img', (recons+1)/2, 1)

    logging.info(f"Test Loss: {loss_meter.avg} ReconsLoss: {recons_meter.avg} KLD: {kld_meter.avg}")
    writer.add_scalar('Test/loss', loss_meter.avg)
    writer.add_scalar('Test/recons_loss', recons_meter.avg)
    writer.add_scalar('Test/kld', kld_meter.avg)

    # FLOPs 계산 및 기록
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 모델 입력 크기에 맞게 조정
    flops = torchprofile.profile_macs(network, dummy_input)
    writer.add_scalar('FLOPs', flops)

    return loss_meter.avg
