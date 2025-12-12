import torch
import torch.nn.functional as F

EPS=torch.finfo(torch.float32).eps
MIN=torch.finfo(torch.float32).tiny

import numpy as np

from torch_utils import persistence
from scipy.stats import dirichlet

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
@persistence.persistent_class
class SDDLoss:
    def __init__(self, eta=None, sigmoid_start=None, sigmoid_end=None, sigmoid_power=None, Scale=None, Shift=None, T=200, epsilon_t=1e-5, lossType='HKD'):
        self.eta = eta
        self.lambda_penalty = 2e-6
        self.sigmoid_start = sigmoid_start
        self.sigmoid_end = sigmoid_end
        self.sigmoid_power = sigmoid_power
        self.Scale = Scale
        self.Shift = Shift
        self.T = T
        self.lossType = lossType
        self.epsilon_t = epsilon_t
        self.min = torch.finfo(torch.float32).tiny
        self.eps = torch.finfo(torch.float32).eps
    def __call__(self, net, images, model, labels=None, augment_pipe=None, ):

        if 1:
            rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
            rnd_position = 1 + rnd_uniform * (self.epsilon_t - 1)
            logit_alpha = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position**self.sigmoid_power)
            rnd_position_previous = rnd_position*0.95
            logit_alpha_previous = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position_previous**self.sigmoid_power)

            alpha = logit_alpha.sigmoid()
            alpha_previous = logit_alpha_previous.sigmoid()

            delta  = (logit_alpha_previous.to(torch.float64).sigmoid()-logit_alpha.to(torch.float64).sigmoid()).to(torch.float32)
        else:
            rnd_position = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
            rnd_position_previous = rnd_position*0.95
            
            alpha = (torch.tensor(2e-6,device=images.device).log().to(torch.float64)*rnd_position).exp()
            alpha_previous = (torch.tensor(1e-6,device=images.device).log().to(torch.float64)*rnd_position_previous).exp()
            
            delta = (alpha_previous-alpha).to(torch.float32)
            
            if 0: 
                logit_alpha_previous = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position**self.sigmoid_power)
                alpha_previous = logit_alpha_previous.sigmoid()
            
                alpha = alpha_previous*0.95
                delta = alpha_previous*0.05
            
            logit_alpha = alpha.logit().to(torch.float32)
            alpha = alpha.to(torch.float32)

        eta= torch.ones([images.shape[0], 1, 1, 1], device=images.device) * self.eta
        
        x0, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        x0 = ((x0+1.0)/2.0).clamp(0,1) * self.Scale + self.Shift

        log_u = self.log_gamma (self.eta * alpha * x0)
        log_v = self.log_gamma (self.eta - self.eta * alpha * x0)

        logit_x_t = (log_u - log_v).to(images.device) 

        diffs_sq = torch.zeros_like(logit_x_t[:, 0, :, :])
        num_channels = logit_x_t.shape[1]
        for i in range(num_channels - 1):
            diff = logit_x_t[:, i+1, :, :] - logit_x_t[:, i, :, :]
            # diff = torch.log(torch.abs(logit_x_t[:, i+1, :, :]) + self.eps) - torch.log(torch.abs(logit_x_t[:, i, :, :]) + self.eps)
            # diff = logit_x_t[:, i+1, :, :] - 2 * logit_x_t[:, i, :, :] + logit_x_t[:, i-1, :, :]
            diffs_sq += diff**2


        penalty =  (self.lambda_penalty * diffs_sq).unsqueeze(1)
        E_exp_penalty = torch.mean(torch.exp(penalty))
        
        logit_x_t = (logit_x_t / E_exp_penalty)

        xmin = self.Shift
        xmax = self.Shift + self.Scale
        xmean = self.Shift+self.Scale/2.0
        E1 = 1.0/(self.eta*alpha*self.Scale)*((self.eta * alpha * xmax).lgamma() - (self.eta * alpha * xmin).lgamma())
        E2 = 1.0/(self.eta*alpha*self.Scale)*((self.eta-self.eta * alpha * xmin).lgamma() - (self.eta-self.eta * alpha * xmax).lgamma())
        E_logit_x_t =  E1 - E2

        V1 = 1.0/(self.eta*alpha*self.Scale)*((self.eta * alpha * xmax).digamma() - (self.eta * alpha * xmin).digamma())
        V2 = 1.0/(self.eta*alpha*self.Scale)*((self.eta-self.eta * alpha * xmin).digamma() - (self.eta-self.eta * alpha * xmax).digamma())

        if 1:
            grids = (torch.arange(0,101,device=images.device)/100)*self.Scale+self.Shift
            alpha_x = alpha[:,:,0,0]*grids.unsqueeze(0)
            V3 =  ((self.eta * alpha_x).digamma())**2
            V3[:,0] = (V3[:,0]+V3[:,-1])/2
            V3 = V3[:,:-1]
            V3 = (V3.mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E1**2).clamp(0)
            V4 = ((self.eta - self.eta*alpha_x).digamma())**2
            V4[:,0] = (V4[:,0]+V4[:,-1])/2
            V4 = V4[:,:-1]
            V4 = (V4.mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E2**2).clamp(0)

        std_logit_x_t = (V1+V2+V3+V4).sqrt()        
        normalized_input = (logit_x_t - E_logit_x_t) / (std_logit_x_t)
        logit_x0_hat = net(normalized_input, logit_alpha, labels, augment_labels=augment_labels)
        x0_hat=torch.sigmoid(logit_x0_hat)* self.Scale + self.Shift
        loss = self.compute_loss(x0, x0_hat,alpha, alpha_previous,eta,delta)
        return loss


    def compute_penalty_integral(self, alpha_p, beta_p, alpha_q, beta_q, num_samples=100):
        delta_p = self.delta if hasattr(self, 'delta') else 1e-6
        delta_q = self.delta if hasattr(self, 'delta') else 1e-6
        def sample_sdd(alpha, delta, num_samples):

            alpha_evt = alpha.permute(0, 2, 3, 1).contiguous()
            B, H, W, C = alpha_evt.shape
            S = num_samples

            alpha_evt = alpha_evt.clamp_min(self.eps)
            dir_dist = torch.distributions.Dirichlet(alpha_evt)
            dir_samples = dir_dist.sample((S,))
            delta_terms = torch.zeros((S, B, H, W), device=alpha.device, dtype=alpha.dtype)
            for j in range(C - 1):
                diff = dir_samples[..., j + 1] - dir_samples[..., j]
                delta_terms += diff**2

            temperature = 0.02
            log_weights = -delta * delta_terms / temperature

            log_weights_max = torch.amax(log_weights, dim=0, keepdim=True)
            shifted_log_weights = log_weights - log_weights_max
            sum_exp = torch.sum(torch.exp(shifted_log_weights), dim=0, keepdim=True)
            log_sum_exp = torch.log(sum_exp) + log_weights_max
            log_weights_normalized = log_weights - log_sum_exp
            weights = torch.exp(log_weights_normalized)

            return weights, delta_terms, dir_samples

        torch.manual_seed(42)
        weights_p, delta_px, _ = sample_sdd(alpha_p, delta_p, num_samples)
        weights_q, delta_qx, _ = sample_sdd(alpha_q, delta_q, num_samples)
        torch.manual_seed(torch.seed())

        log_penalty_p = torch.log(torch.sum(weights_p * delta_px, dim=0))
        log_penalty_q = torch.log(torch.sum(weights_q * delta_qx, dim=0))
        
        penalty_diff = torch.exp(log_penalty_p) - torch.exp(log_penalty_q)
        return penalty_diff.mean()


    
    def compute_loss(self, x0, x0_hat, alpha, alpha_previous, eta, delta):

        x0 = x0.clamp(self.Shift, self.Shift + self.Scale)
        x0_hat = x0_hat.clamp(self.Shift, self.Shift + self.Scale)
        
        alpha_p = eta*delta*x0
        beta_p = eta-eta*alpha_previous*x0 
        alpha_q = eta*delta*x0_hat
        beta_q  = eta-eta*alpha_previous*x0_hat 

        _alpha_p = eta*alpha*x0
        _beta_p  = eta-eta*alpha*x0
        _alpha_q = eta*alpha*x0_hat
        _beta_q  = eta-eta*alpha*x0_hat

        penalty_integral = self.compute_penalty_integral(alpha_p, beta_p, alpha_q, beta_q)

        KLUB_conditional = (self.KL_gamma(alpha_q,alpha_p).clamp(0)\
                                + self.KL_gamma(beta_q,beta_p).clamp(0)\
                                - self.KL_gamma(alpha_q+beta_q,alpha_p+beta_p).clamp(0)-penalty_integral.clamp(0)).clamp(0)
        KLUB_marginal = (self.KL_gamma(_alpha_q,_alpha_p).clamp(0)\
                            + self.KL_gamma(_beta_q,_beta_p).clamp(0)\
                            - self.KL_gamma(_alpha_q+_beta_q,_alpha_p+_beta_p).clamp(0)-penalty_integral.clamp(0)).clamp(0)
        KLUB_conditional_AS = (self.KL_gamma(alpha_p,alpha_q).clamp(0)\
                                    + self.KL_gamma(beta_p,beta_q).clamp(0)\
                                    - self.KL_gamma(alpha_p+beta_p,alpha_q+beta_q).clamp(0)-penalty_integral.clamp(0)).clamp(0)
        KLUB_marginal_AS = (self.KL_gamma(_alpha_p,_alpha_q).clamp(0)\
                                + self.KL_gamma(_beta_p,_beta_q).clamp(0)\
                                - self.KL_gamma(_alpha_p+_beta_p,_alpha_q+_beta_q).clamp(0)-penalty_integral.clamp(0)).clamp(0)

        HPD1 = (self.HPD(alpha_p,alpha_q).clamp(0) - penalty_integral.clamp(0)).clamp(0)
        HPD2 = (self.HPD(_alpha_p,_alpha_q).clamp(0) - penalty_integral.clamp(0)).clamp(0)

        loss_dict = {
            'KLUB': (0.96 * KLUB_conditional + 0.04 * KLUB_marginal),
            'HKD': ( 0.7 * HPD1 + 0.97 * KLUB_conditional + 0.03 * KLUB_marginal),
            'ELBO': 0.99 * KLUB_conditional_AS + 0.01 * KLUB_marginal_AS,
        }

        if self.lossType not in loss_dict:
            raise NotImplementedError("Loss type not implemented")
        
        return loss_dict[self.lossType]
    
    def HPD(self,alpha_p,alpha_q):
        a = 1.8
        b = a/(a-1)
        c = 32
        sbeta = torch.ones((1, c)).cuda()
        F1_1 = torch.sum(torch.lgamma(a * alpha_p + 0.5*sbeta), dim=1, keepdim=True) - torch.lgamma(torch.sum((a * alpha_p + 0.5*sbeta), dim=1, keepdim=True))
        F2_1 = torch.sum(torch.lgamma(b * alpha_q + 0.5*sbeta), dim=1, keepdim=True) - torch.lgamma(torch.sum((b * alpha_q + 0.5*sbeta), dim=1, keepdim=True))
        F3_1 = torch.sum(torch.lgamma(alpha_p + alpha_q + 0.5*sbeta), dim=1, keepdim=True) - torch.lgamma(torch.sum((alpha_p + alpha_q + 0.5*sbeta), dim=1, keepdim=True))

        hd1 = 1/a * F1_1 + 1/b * F2_1 - F3_1

        a = b
        b = a/(a-1)
        
        F1_2 = torch.sum(torch.lgamma(a * alpha_p + 0.5*sbeta), dim=1, keepdim=True) - torch.lgamma(torch.sum((a * alpha_p + 0.5*sbeta), dim=1, keepdim=True))
        F2_2 = torch.sum(torch.lgamma(b * alpha_q + 0.5*sbeta), dim=1, keepdim=True) - torch.lgamma(torch.sum((b * alpha_q + 0.5*sbeta), dim=1, keepdim=True))
        F3_2 = torch.sum(torch.lgamma(alpha_p + alpha_q + 0.5*sbeta), dim=1, keepdim=True) - torch.lgamma(torch.sum((alpha_p + alpha_q + 0.5*sbeta), dim=1, keepdim=True))

        hd2 = 1/a * F1_2 + 1/b * F2_2 - F3_2

        hd = 1/2 * (hd1+hd2)

        return hd
      
    def log_gamma(self, alpha):
        return torch.log(torch._standard_gamma(alpha.to(torch.float32)).clamp(MIN))
    def KL_gamma(self, alpha_p, alpha_q, beta_p=None, beta_q=None):
        KL = (alpha_p-alpha_q)*torch.digamma(alpha_p)-torch.lgamma(alpha_p)+torch.lgamma(alpha_q)
        if beta_p is not None and beta_q is not None:
            KL = KL + alpha_q*(torch.log(beta_p)-torch.log(beta_q))+alpha_p*(beta_q/beta_p-1.0)
        return KL
    
    
