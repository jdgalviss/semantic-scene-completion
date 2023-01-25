import math
import torch
import torch.nn as nn


def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

class SNConv2WithActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConv2WithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
    def forward(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            return self.activation(x)
        return x

class Discriminator2D(nn.Module):
    def __init__(self, nf_in, nf, patch_size, image_dims, patch, use_bias, disc_loss_type='vanilla'):
        nn.Module.__init__(self)
        self.use_bias = use_bias
        approx_receptive_field_sizes = [4, 10, 22, 46, 94, 190, 382, 766]
        num_layers = len(approx_receptive_field_sizes)
        if patch:
            for k in range(len(approx_receptive_field_sizes)):
                if patch_size < approx_receptive_field_sizes[k]:
                    num_layers = k
                    break
        assert(num_layers >= 1)
        self.patch = patch
        self.nf = nf
        dim = min(image_dims[0], image_dims[1])
        num = int(math.floor(math.log(dim, 2)))
        num_layers = min(num, num_layers)
        activation = None if num_layers == 1 else torch.nn.LeakyReLU(0.2, inplace=True)
        self.discriminator_net = torch.nn.Sequential(
            SNConv2WithActivation(nf_in, 2*nf, 4, 2, 1, activation=activation, bias=self.use_bias),
        )
        if num_layers > 1:
            activation = None if num_layers == 2 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p1', SNConv2WithActivation(2*nf, 4*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        if num_layers > 2:
            activation = None if num_layers == 3 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p2', SNConv2WithActivation(4*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        for k in range(3, num_layers):
            activation = None if num_layers == k+1 else torch.nn.LeakyReLU(0.2, inplace=True)
            self.discriminator_net.add_module('p%d' % k, SNConv2WithActivation(8*nf, 8*nf, 4, 2, 1, activation=activation, bias=self.use_bias))
        self.final = None
        if not patch or disc_loss_type != 'hinge': #hack
            self.final = torch.nn.Conv2d(nf*8, 1, 1, 1, 0)        
        num_params = count_num_model_params(self.discriminator_net)
        print('#params discriminator', count_num_model_params(self.discriminator_net))
        
        self.compute_valid = None
        if patch:
            self.compute_valid = torch.nn.Sequential(
                torch.nn.AvgPool2d(4, stride=2, padding=1),
            )
            for k in range(1, num_layers):
                self.compute_valid.add_module('p%d' % k, torch.nn.AvgPool2d(4, stride=2, padding=1))
    
    def compute_valids(self, valid):
        if self.compute_valid is None:
            return None
        valid = self.compute_valid(valid)
        return valid

    def forward(self, x, alpha=None):
        for k in range(len(self.discriminator_net)-1):
            x = self.discriminator_net[k](x)
        x = self.discriminator_net[-1](x) 
        
        if self.final is not None:
            x = self.final(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x



class GANLoss:

    def __init__(self, loss_type):
        self.compute_generator_loss = GANLoss.compute_generator_loss_wasserstein
        self.compute_discriminator_loss = None
        if loss_type == 'hinge':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_hinge
        elif loss_type == 'vanilla':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_vanilla
            self.compute_generator_loss = GANLoss.compute_generator_loss_vanilla
        elif loss_type == 'wgan':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_wasserstein
        elif loss_type == 'wgan_gp':
            self.compute_discriminator_loss = GANLoss.compute_discriminator_loss_wasserstein_gp

    @staticmethod
    def compute_discriminator_loss_vanilla(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        d_real = ref_disc(in_real, alpha)
        d_fake = ref_disc(in_fake, alpha)
        if weight is not None:
            d_real = d_real * weight.view(d_real.shape)
            d_fake = d_fake * weight.view(d_fake.shape)        
        if valid is not None:
            d_real = d_real[valid]
            d_fake = d_fake[valid]
        real_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_real, torch.ones(d_real.shape).cuda()*label_smoothing_factor, reduction='none')
        real_loss = torch.mean(real_loss, 1)
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_fake, torch.zeros(d_fake.shape).cuda(), reduction='none')
        fake_loss = torch.mean(fake_loss, 1)
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)
        
    @staticmethod
    def compute_generator_loss_vanilla(ref_disc, in_fake, alpha=1):
        d_fake = ref_disc(in_fake, alpha)
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(d_fake, torch.ones(d_fake.shape).cuda())
        return fake_loss
        
    @staticmethod
    def compute_generator_loss_wasserstein(ref_disc, in_fake, alpha=1):
        return -torch.mean(ref_disc(in_fake, alpha))

    @staticmethod
    def compute_discriminator_loss_wasserstein(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        d_real = ref_disc(in_real, alpha)
        d_fake = ref_disc(in_fake, alpha)
        if weight is not None:
            d_real = d_real * weight.view(d_real.shape)
            d_fake = d_fake * weight.view(d_fake.shape)  
        if valid is not None:
            d_real = d_real[valid]
            d_fake = d_fake[valid]
        real_loss = -torch.mean(d_real,1)
        fake_loss = torch.mean(d_fake,1)
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)

    @staticmethod
    def compute_discriminator_loss_hinge(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        d_real = ref_disc(in_real, alpha)
        d_fake = ref_disc(in_fake, alpha)
        if weight is not None:
            d_real = d_real * weight.view(d_real.shape)
            d_fake = d_fake * weight.view(d_fake.shape)         
        if valid is not None:
            d_real = d_real[valid]
            d_fake = d_fake[valid]
        real_loss = torch.mean(torch.nn.functional.relu(1. - d_real), 1)
        fake_loss = torch.mean(torch.nn.functional.relu(1. + d_fake), 1)
        return real_loss, fake_loss, torch.Tensor([0]).requires_grad_(True).cuda(in_real.device.index)

    @staticmethod
    def compute_discriminator_loss_wasserstein_gp(ref_disc, in_real, in_fake, valid, weight, val_mode=False, label_smoothing_factor=1, alpha=1):
        if not val_mode:
            real_loss, fake_loss, _ = GANLoss.compute_discriminator_loss_wasserstein(ref_disc, in_real, in_fake, valid, weight, val_mode, alpha)
            return real_loss, fake_loss, GANLoss.compute_gradient_penalty(ref_disc, in_real, in_fake)
        else:
            return GANLoss.compute_discriminator_loss_wasserstein(ref_disc, in_real, in_fake, valid, weight, val_mode, alpha)

    @staticmethod
    def compute_gradient_penalty(ref_disc, in_real, in_fake):
        # Calculate interpolation
        if len(in_real.shape) == 5:
            alpha = torch.rand(in_real.shape[0], 1, 1, 1, 1)
        else:
            alpha = torch.rand(in_real.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(in_real)
        alpha = alpha.cuda(in_real.device.index)
        interpolated = (alpha * in_real + (1 - alpha) * in_fake).requires_grad_(True)

        # Calculate probability of interpolated examples
        prob_interpolated = ref_disc(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).cuda(in_real.device.index), create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(in_real.shape[0], -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()
