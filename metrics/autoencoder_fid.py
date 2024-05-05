import os
import glob
from tqdm import tqdm
from glob import glob
import utils.global_v as glv
from cleanfid import fid as Fid

import torch
import random
import numpy as np
from scipy import linalg

from PIL import Image
import models.svae_models.esvae as esvae
from utils.datasets import load_dataset_snn


def get_autoencoder_frechet_distance(model, dataset, device, dir_name, num_gen=5000):
    def sample_from_vae(batch_size):
        sampled_x, _ = model.sample(batch_size)
        return sampled_x

    score = compute_autoencoder_frechet_distance(dir_name, gen=sample_from_vae, dataset_name=dataset,
            num_gen=num_gen, batch_size=32, device=device)

    return score

def get_autoencoder_frechet_distance_ann(dir_name, model, dataset, device, num_gen=5000):
    def sample_from_vae(batch_size):
        sampled_x = model.sample(batch_size, device)
        return sampled_x

    score = compute_autoencoder_frechet_distance(dir_name, gen=sample_from_vae, dataset_name=dataset,
            num_gen=num_gen, batch_size=32, device=device)

    return score

def compute_autoencoder_frechet_distance(dir_name, gen, dataset_name, num_gen=5000, batch_size=32,
                                        device=torch.device("cuda")):
    if dataset_name.lower() == 'mnist':     
        in_channels = 1 
        latent_dim=64
        feat_model = esvae.ESVAE(device=device, distance_lambda=glv.network_config['distance_lambda'], 
                                    mmd_type=glv.network_config['mmd_type']).to(device)
    elif dataset_name.lower() == 'fashion':
        in_channels = 1
        latent_dim=64
        feat_model = esvae.ESVAE(device=device, distance_lambda=glv.network_config['distance_lambda'], 
                                    mmd_type=glv.network_config['mmd_type']).to(device)
    elif dataset_name.lower() == 'celeba':
        in_channels = 3
        latent_dim = 128
        feat_model = esvae.ESVAE(device=device, distance_lambda=glv.network_config['distance_lambda'], 
                                    mmd_type=glv.network_config['mmd_type']).to(device)
    elif dataset_name.lower() == 'cifar10':
        in_channels = 3
        latent_dim = 128
        feat_model = esvae.ESVAE(device=device, distance_lambda=glv.network_config['distance_lambda'], 
                                    mmd_type=glv.network_config['mmd_type']).to(device)
    elif dataset_name.lower() == 'miad_metal_welding':
        in_channels = 3
        latent_dim = 32
        feat_model = esvae.ESVAE(device=device, distance_lambda=glv.network_config['distance_lambda'], 
                                    mmd_type=glv.network_config['mmd_type']).to(device)
    else:
        raise ValueError()
    
    stat_name = f'./metrics/stats/{dataset_name.lower()}_test_{latent_dim}.npz'
    checkpoint = os.path.join(dir_name, "checkpoint.pth")
    if not os.path.exists(stat_name):
        make_custom_stats(stat_name, dataset_name, checkpoint, glv.network_config["data_path"],
                        batch_size, device, glv.network_config['input_size'])
    stats = np.load(stat_name)
    ref_mu, ref_sigma = stats["mu"], stats["sigma"]
    feat_model.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))

    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    print("compute Frechet Autoencoder Distance")
    for idx in tqdm(range(num_iters)):
        with torch.no_grad():
            # genearted image is in range [-1,1]
            # no need to resize
            img_batch = gen(batch_size).unsqueeze(-1).repeat(1, 1, 1, 1, 16) 
            feat = feat_model.encode(img_batch)
        l_feats = feat[0].detach().cpu().numpy()
    np_feats = np.concatenate(l_feats)
    
    mu = np.mean(np_feats)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid

def make_custom_stats(outf, dataset_name, checkpoint, data_path,
                        batch_size, device, input_size):
    if os.path.exists(outf):
        msg = f"The statistics file {outf} already exists. "
        raise Exception(msg)
    
    if dataset_name == 'miad_metal_welding':
        _, test_loader = load_dataset_snn.load_MIAD_metal_welding(data_path, batch_size, input_size)
        feat_model = esvae.ESVAE(device=device, distance_lambda=glv.network_config['distance_lambda'], 
                                    mmd_type=glv.network_config['mmd_type']).to(device)
    else:
        raise ValueError()
    feat_model.load_state_dict(torch.load(checkpoint))
    
    l_feats = []
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            img = img.unsqueeze(-1).repeat(1, 1, 1, 1, 16)  # (N, C, H, W, T)
            latent = feat_model.encode(img.to(device))
            l_feats = latent[0].detach().cpu().numpy()
    np_feats = np.concatenate(l_feats)
    mu = np.mean(np_feats)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
    

#device = torch.device("cuda")
#stat_name = f'./metrics/stats/{dataset_name.lower()}_test_{latent_dim}'
#checkpoint = f'./data/ESVAE/checkpoint/MIAD_metal_welding/esvae_lr-0.000384_lambda-0.001_loss_func-mmd_mmd_type-rbf_sample_layer_lr_times-10-latent_dim-32/checkpoint.pth'
#make_custom_stats(stat_name, dataset_name, checkpoint, latent_dim, batch_size, device, input_size)
# def make_custom_stats(stat_name, dataset_name, checkpoint, latent_dim=32, 
#                         batch_size=32, device=torch.device("cuda")):
#     outname = f'{stat_name}.npz'
#     outf = os.path.join(outname)
#     # if os.path.exists(outf):
#     #     msg = f"The statistics file {stat_name} already exists. "
#     #     raise Exception(msg)
    
    
#     if dataset_name == 'mnist':     
#         _, test_loader = load_dataset_ann.load_mnist(batch_size)
#         in_channels = 1 
#         feat_model = ann_ae.AE(in_channels, latent_dim).to(device)
#     elif dataset_name == 'fashion':
#         _, test_loader = load_dataset_ann.load_fashionmnist(batch_size)
#         in_channels = 1
#         feat_model = ann_ae.AE(in_channels, latent_dim).to(device)
#     elif dataset_name == 'celeba':
#         _, test_loader = load_dataset_ann.load_celeba(batch_size)
#         in_channels = 3
#         feat_model = ann_ae.AELarge(in_channels, latent_dim).to(device)
#     elif dataset_name == 'cifar10':
#         _, test_loader = load_dataset_ann.load_cifar10(batch_size)
#         in_channels = 3
#         feat_model = ann_ae.AE(in_channels, latent_dim).to(device)
#     elif dataset_name == 'MIAD_metal_welding':
#         _, test_loader = load_dataset_ann.load_MIAD_metal_welding(batch_size)
#         in_channels = 3
#         feat_model = ann_ae.AE(in_channels, latent_dim).to(device)
#     else:
#         raise ValueError()

#     feat_model.load_state_dict(torch.load(checkpoint))
    
#     l_feats = []
#     with torch.no_grad():
#         for img, label in tqdm(test_loader):
#             img = img.unsqueeze(-1).repeat(1, 1, 1, 1, 16)  # (N, C, H, W, T)
#             latent = feat_model.encode(img.to(device))
#             l_feats.append(latent.detach().cpu().numpy())
#     np_feats = np.concatenate(l_feats)
#     print(np_feats)
#     mu = np.mean(np_feats, axis=0)
#     sigma = np.cov(np_feats, rowvar=False)
#     print(f"saving custom stats to {outf}")
#     np.savez_compressed(outf, mu=mu, sigma=sigma)

"""
Compute the FID score given the mu, sigma of two sets
"""
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Danica J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    print(mu1.shape, mu2.shape)
    print(sigma1.shape, sigma2.shape)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)