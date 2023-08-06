import math
import types

import torch
from torch import nn
from torch.nn import functional as F

def init_diag(self,
              config):
    def register(n, x):
        return self.register_parameter(n, nn.Parameter(x))

    init_val = config.q_init_logvar
    init_log_w = init_val * torch.ones_like(self.weight)
    register('_diag_w', init_log_w)

    if self.bias is not None:
        init_log_b = init_val * torch.ones_like(self.bias)
        register('_diag_b', init_log_b)

def set_prior_logvar(module, config):
    out_dim = module.weight.shape[0] \
    if isinstance(module, nn.Linear) \
    else module.weight.shape[1]

    if not config.scale_prior: 
        out_dim = 1

    module.p_logvar = -math.log(config.prior_precision*out_dim)
    
def get_variational_step(opt, num_points,num_variational_samples = 1):
    def step(self, x, y, loss_func):
        opt.zero_grad()
        l = 0
        with torch.enable_grad():
            for _ in range(num_variational_samples):
                l += loss_func(self(x), y)
            l /= num_variational_samples
            kl = sum(m.kl()
                     for m in self.modules()
                     if hasattr(m, "kl"))
            neg_elbo = l + 1 / num_points * kl
        neg_elbo.backward()
        opt.step()

        return {'elbo': neg_elbo.neg().item(),
                'kl/n': kl.item() / num_points,
                'neg_data_term': l.item()}

    return step

def lrt(self, input_, mean):
    (x,) = input_

    var_w = self._diag_w.exp()
    var_b = self._diag_b.exp() \
        if self.bias is not None \
        else None

    if isinstance(self, nn.Linear):
        mf_var = F.linear(x.pow(2),
                          var_w,
                          var_b)

    elif isinstance(self, nn.Conv2d):
        mf_var = F.conv2d(x.pow(2),
                          var_w,
                          var_b,
                          self.stride,
                          self.padding,
                          self.dilation,
                          self.groups)
        mf_var = mf_var.clamp(min=1e-16)
    else:
        raise NotImplementedError()

    mf_noise = mf_var.sqrt() * torch.randn_like(mf_var,
                                                requires_grad=False)

    return mean + mf_noise

def cmfvi(model,
          config,
          num_points):
    def modify(module):

        if isinstance(module, nn.Dropout):
            module.p = 0.

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            
            set_prior_logvar(module, config)
            init_diag(module, config)
            module.register_forward_hook(lrt)
            kl = get_collapsed_kl(config)
            module.kl = types.MethodType(kl, module)

    model.inference_type = 'VI'
    model.num_test_samples = config.num_test_samples
    model.apply(modify)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.step = types.MethodType(get_variational_step(opt, num_points), model)

    return model

def get_collapsed_kl(config):
        
    def _kl(self):

        def f(m1, v1, m2, v2):
            v2log = v2.log() \
                if isinstance(v2, torch.Tensor) \
                else math.log(v2)

            term1 = 0.5 * (v2log - v1.log())

            alpha_reg = torch.tensor(config.alpha)

            log_alpha = -0.5 * torch.log(alpha_reg) * v1.numel()
            term2 = (v1 + alpha_reg * m1.pow(2)) / (2 * v2)

            return (term1 + term2 - 0.5).sum() + log_alpha

        kl_w = f(self.weight,
                             self._diag_w.exp(),
                             0,
                             math.exp(self.p_logvar))
        if self.bias is not None:
            kl_b = f(self.bias,
                                 self._diag_b.exp(),
                                 0,
                                 math.exp(self.p_logvar))
        else:
            kl_b = 0

        return kl_w + kl_b

    return _kl
