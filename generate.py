import os
import sys
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch.distributions import Beta
import torch.nn.functional as F

import math
import functools
from torch.nn.functional import logsigmoid

import json

from torch.nn.functional import logsigmoid

MIN = torch.finfo(torch.float32).tiny
EPS = torch.finfo(torch.float32).eps

def sigma_sigma(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    return (0.5 * beta_d * (t ** 2) + beta_min * t).expm1().sqrt()

def alpha_alpha(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    return (-0.5 * beta_d * (t ** 2) - beta_min * t).exp()   

def log_alpha_log_alpha(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    return -0.5 * beta_d * (t ** 2) - beta_min * t 

def get_sigma(y, sigma_max):
    sigma_min = torch.tensor([0.01]).to(y.device)
    sigma_max = torch.tensor([sigma_max]).to(y.device)
    ts = torch.linspace(1.0, 1e-3, 1000).to(y.device)
    ss = sigma_min * (sigma_max / sigma_min).to(y.device) ** ts
    return ss[y]

def SDD_sampler(
    net, latents, class_labels=None,randn_like=torch.randn_like,
    num_steps=None, alpha_min=None, eta=None, sigmoid_start=None, sigmoid_end=None, sigmoid_power=None, start_step=None,Scale=None,Shift=None,
):
    lambda_penalty = 2e-6

    if num_steps>350:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)

        t_steps = 1-step_indices / (num_steps - 1)*(1-1e-5)
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        logit_alpha = sigmoid_start + (sigmoid_end-sigmoid_start) * (t_steps**sigmoid_power)
        alpha = logit_alpha.sigmoid()
    
    else:
        step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
        log_pi = (sigmoid_start + (sigmoid_end - sigmoid_start) * (torch.tensor(1, dtype=torch.float64, device=latents.device)**sigmoid_power)).sigmoid().log() / (num_steps)
        alpha = (step_indices*log_pi).exp()
        alpha = torch.flip(alpha, [0])
        logit_alpha = alpha.logit()

    alpha_min = alpha_min if alpha_min is not None else alpha[0]


    if 1:
        log_u = log_gamma( (eta * alpha_min * latents).to(torch.float32) ).to(torch.float64)
        log_v = log_gamma( (eta - eta * alpha_min * latents).to(torch.float32) ).to(torch.float64)
        x_next = (log_u - log_v).to(latents.device)
    else:
        x_next = torch.logit( alpha_min*latents.to(torch.float64) ).to(latents.device)

    diffs_sq = torch.zeros_like(x_next[:, 0, :, :])  

    num_channels = x_next.shape[1]

    for i in range(num_channels - 1):
        diff = x_next[:, i+1, :, :] - x_next[:, i, :, :]
        # diff = torch.log(torch.abs(x_next[:, i+1, :, :])  + EPS) - torch.log(torch.abs(x_next[:, i, :, :]) + EPS)
        # diff = x_next[:, i+1, :, :] - 2 * x_next[:, i, :, :] + x_next[:, i-1, :, :]
        diffs_sq += diff**2

    penalty = lambda_penalty * diffs_sq

    E_exp_penalty = torch.mean(torch.exp(penalty)).clamp_min(EPS)

    x_next = (x_next/E_exp_penalty)  
    ims = []
    im_xhats = []

    ims.append( ((torch.sigmoid(x_next)/(alpha_min)-Shift)/Scale-0.5)/0.5)
    im_xhats.append(((latents - Shift) / Scale-0.5)/0.5) 

    for i, (logit_alpha_cur,logit_alpha_next) in enumerate(zip(logit_alpha[:-1], logit_alpha[1:])): 
        x_cur = x_next
        alpha_cur = logit_alpha_cur.sigmoid()
        alpha_next = logit_alpha_next.sigmoid()

        log_alpha_cur = logsigmoid(logit_alpha_cur)

        xmin = Shift
        xmax = Shift + Scale
        xmean = Shift+Scale/2.0

        E1 = 1.0/(eta*alpha_cur*Scale)*((eta * alpha_cur * xmax).lgamma() - (eta * alpha_cur * xmin).lgamma())
        E2 = 1.0/(eta*alpha_cur*Scale)*((eta-eta * alpha_cur * xmin).lgamma() - (eta-eta * alpha_cur * xmax).lgamma())
        E_logit_x_t =  E1 - E2

        V1 = 1.0/(eta*alpha_cur*Scale)*((eta * alpha_cur * xmax).digamma() - (eta * alpha_cur * xmin).digamma())
        V2 = 1.0/(eta*alpha_cur*Scale)*((eta-eta * alpha_cur * xmin).digamma() - (eta-eta * alpha_cur * xmax).digamma())

        grids = (torch.arange(0,101,device=latents.device)/100 +0.5/100)*Scale+Shift
        grids = grids[:-1]
        alpha_x = alpha_cur*grids 
        if 1:
            grids = (torch.arange(0,101,device=latents.device)/100)*Scale+Shift
            alpha_x = alpha_cur*grids 
            V3 = ((eta * alpha_x).digamma())**2
            V3[0] = (V3[0]+V3[-1])/2
            V3 = V3[:-1]
            V3 = (V3.mean()- E1**2).clamp(0)   
            V4 = ((eta - eta*alpha_x).digamma())**2
            V4[0] = (V4[0]+V4[-1])/2
            V4 = V4[:-1]
            V4 = (V4.mean()- E2**2).clamp(0)
            
        else:
            V3 = (( ((eta * alpha_x).digamma())**2).mean()- E1**2).clamp(0)           
            V4 = (( ((eta - eta*alpha_x).digamma())**2).mean()- E2**2).clamp(0)

        std_logit_x_t = (V1+V2+V3+V4).sqrt()
        logit_x_t = x_cur
        norm_inp = (logit_x_t - E_logit_x_t) / (std_logit_x_t + EPS)
        if class_labels is None:
            logit_x0_hat = net(norm_inp, logit_alpha_cur).to(torch.float64)
        else:    
            logit_x0_hat = net(norm_inp, logit_alpha_cur, class_labels).to(torch.float64)

        x0_hat = torch.sigmoid(logit_x0_hat)* Scale + Shift
        
        im_xhats.append(((x0_hat - Shift) / Scale-0.5)/0.5) 
        
        alpha_reverse = (eta*alpha_next-eta*alpha_cur)*x0_hat
        beta_reverse = eta-eta*alpha_next*x0_hat

        log_u = log_gamma(alpha_reverse.to(torch.float32)).to(torch.float64)
        log_v = log_gamma(beta_reverse.to(torch.float32)).to(torch.float64)
        p = (log_u - log_v).to(latents.device)

        concatenated = torch.cat((x_cur.unsqueeze(-1), (p).unsqueeze(-1), (x_cur+p).unsqueeze(-1)), dim=4)
        x_next = torch.logsumexp(concatenated, dim=4)    
        
        ims.append( ((torch.sigmoid(x_next)/(alpha_next)-Shift)/Scale-0.5)/0.5)
        
        
    if 1: 
        out = (x0_hat- Shift) / Scale
        out1 = ((torch.sigmoid(x_next)/(alpha_next)- Shift) / Scale) 
    
    return (out-0.5)/0.5, (out1-0.5)/0.5    


def log_gamma(alpha):
    return torch.log(torch._standard_gamma(alpha))


def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) 

    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): 
        x_cur = x_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=200, show_default=True)

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    
    directory_path = '/'.join(network_pkl.split('/')[:-1])

    jason_file_name = "training_options.json"

    json_file_path = directory_path + '/' + jason_file_name

    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)

    eta = json_data['loss_kwargs']['eta']
    sampler_kwargs['eta'] = eta
    Scale = json_data['loss_kwargs']['Scale']
    sampler_kwargs['Scale']=Scale
    Shift = json_data['loss_kwargs']['Shift']
    sampler_kwargs['Shift']=Shift
    lossType = json_data['loss_kwargs']['lossType']
    
    
    sigmoid_start = json_data['loss_kwargs']['sigmoid_start']
    sampler_kwargs['sigmoid_start']= sigmoid_start
    sigmoid_end = json_data['loss_kwargs']['sigmoid_end']
    sampler_kwargs['sigmoid_end']= sigmoid_end
    sigmoid_power = json_data['loss_kwargs']['sigmoid_power']
    sampler_kwargs['sigmoid_power']= sigmoid_power
    
    print(eta,sigmoid_start,sigmoid_end,sigmoid_power,Scale,Shift,lossType,sampler_kwargs['num_steps'])
    
    saveImage = True
    
    
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root="~/datasets", download=True)
    data = dataset.data / 255.0
    datamean= torch.tensor(data.mean(0)).permute(2, 0, 1)
    
    
    
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        if 0:
            latents = torch.ones([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

            latents = Beta(latents,1.0).sample()* Scale + Shift
        else:
            dm = datamean

            if dm.shape[0] != net.img_channels:
                if net.img_channels == 1 and dm.shape[0] == 3:
                    dm = 0.299*dm[0] + 0.587*dm[1] + 0.114*dm[2]
                    dm = dm.unsqueeze(0)
                elif net.img_channels == 3 and dm.shape[0] == 1:
                    dm = dm.repeat(3,1,1)
                else:
                    dm = dm[:net.img_channels]
            if dm.shape[1] != net.img_resolution or dm.shape[2] != net.img_resolution:
                dm = F.interpolate(dm.unsqueeze(0), size=(net.img_resolution, net.img_resolution), mode='bilinear', align_corners=False)[0]
            latents = (dm*Scale+Shift).expand(batch_size, net.img_channels, net.img_resolution, net.img_resolution).to(device)

        latents = torch.ones([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)*Scale/2+Shift

        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1


        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None} 
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else SDD_sampler 

        images,images1 = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        if saveImage:
            n = len(images_np)
            m = int(np.ceil(np.sqrt(n)))
            w = h = net.img_resolution
            grid = np.zeros((m*h, m*w, 3), dtype=np.uint8)
            for i, i_np in enumerate(images_np):
                x, y = i%m, i//m
                grid[y*h:(y+1)*h, x*w:(x+1)*w, :] = i_np

            image_grid = PIL.Image.fromarray(grid, 'RGB')
            random_number = np.random.rand()
            image_grid.save(f"{network_pkl[0:-4]}_{random_number:.6f}.png")
        
        images_np = (images1 * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        outdir1 = outdir+'_1'
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir1, f'{seed-seed%1000:06d}') if subdirs else outdir1
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        if saveImage:
            n = len(images_np)
            m = int(np.ceil(np.sqrt(n)))
            w = h = net.img_resolution
            grid = np.zeros((m*h, m*w, 3), dtype=np.uint8)
            for i, i_np in enumerate(images_np):
                x, y = i%m, i//m
                grid[y*h:(y+1)*h, x*w:(x+1)*w, :] = i_np

            image_grid = PIL.Image.fromarray(grid, 'RGB')

            random_number = np.random.rand()
            image_grid.save(f"{network_pkl[0:-4]}_{random_number:.6f}_1.png")
        saveImage = False
                

    torch.distributed.barrier()
    dist.print0('Done.')



if __name__ == "__main__":
    
    sys.argv = [
    'generate.py',                            
    '--steps', '200',                        
    '--outdir', 'plots/images06',      
    '--network','SDD-train-runs/00000-cifar10-32x32-uncond-ddpmpp-betadiff-gpus8-batch512-fp32/network-snapshot-200000.pkl',  
    '--seeds', '0-99',                       
    '--batch', '100'                          
]
    main()

