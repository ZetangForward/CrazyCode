import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from projects.custom_llama.models.encdec import Resnet, Resnet1D, Encoder, Decoder, EncoderConvBlock
from models.bottleneck import BottleneckBlock, NoBottleneck, Bottleneck
from models.resnet import ResConv1DBlock, Resnet1D
from models.encdec import Encoder, Decoder, EncoderConvBlock


# helper functions
def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


class VQVAE(nn.Module):
    def __init__(self, config, input_shape, multipliers=None, **block_kwargs):
        super().__init__()
        self.cfg = config.vqvae
        self.commit = self.cfg.commit
        self.spectral = self.cfg.spectral
        self.multispectral = self.cfg.multispectral

        self.downsamples = calculate_strides(self.cfg.strides_t, self.cfg.downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)

        self.sample_length = config.dataset.max_path_nums
        self.x_channels = config.dataset.x_channels
        self.x_shape = (config.dataset.max_path_nums, config.dataset.x_channels)

        if multipliers is None:
            self.multipliers = [1] * self.cfg.levels
        else:
            assert len(multipliers) == self.cfg.levels, "Invalid number of multipliers"
            self.multipliers = multipliers

        self.z_shapes = [(self.x_shape[0] // self.hop_lengths[level],) for level in range(self.cfg.levels)]

        # define encoder and decoder
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs
        
        def encoder(level): 
            return Encoder(
                self.x_channels, self.cfg.emb_width, level + 1,
                self.cfg.downs_t[:level+1], self.cfg.strides_t[:level+1], **_block_kwargs(level)
            )
        
        def decoder(level): 
            return Decoder(
                self.x_channels, self.cfg.emb_width, level + 1,
                self.cfg.downs_t[:level+1], self.cfg.strides_t[:level+1], **_block_kwargs(level)
            )

        for level in range(self.cfg.levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        # define bottleneck
        if self.cfg.use_bottleneck:
            self.bottleneck = Bottleneck(self.cfg.l_bins, self.cfg.emb_width, mu, self.cfg.levels)
        else:
            self.bottleneck = NoBottleneck(self.cfg.levels)    


        

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(
            zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(
                zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(
                x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape),
                        device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x, hps, loss_fn='l1'):
        metrics = {}
        N = x.shape[0]
        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []

        #print("encoder input: ")
        # print(x_in.shape)
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            #t.onnx.export(encoder, x_in,  "encoder_lvl1_verbose.onnx",verbose=True)
            xs.append(x_out[-1])
        #print("encoder output: ")
        # print(xs[0].shape)
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        #print("bottelneck output: ")
        # print(xs_quantised[0].shape)

        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)

            # happens when deploying
            if (x_out.shape != x_in.shape):
                x_out = F.pad(input=x_out, pad=(
                    0, x_in.shape[-1]-x_out.shape[-1]), mode='constant', value=0)

            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)
        #print("decoder output: ")
        # print(x_outs[0].shape)
        # Loss

        def _spectral_loss(x_target, x_out, hps):
            if hps.use_nonrelative_specloss:
                sl = spectral_loss(x_target, x_out, hps) / \
                    hps.bandwidth['spec']
            else:
                sl = spectral_convergence(x_target, x_out, hps)
            sl = t.mean(sl)
            return sl

        def _multispectral_loss(x_target, x_out, hps):
            sl = multispectral_loss(
                x_target, x_out, hps) / hps.bandwidth['spec']
            sl = t.mean(sl)
            return sl

        recons_loss = t.zeros(()).to(x.device)
        spec_loss = t.zeros(()).to(x.device)
        multispec_loss = t.zeros(()).to(x.device)
        x_target = audio_postprocess(x.float(), hps)

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            x_out = audio_postprocess(x_out, hps)
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_spec_loss = _spectral_loss(x_target, x_out, hps)
            this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss

        commit_loss = sum(commit_losses)
        loss = recons_loss + self.spectral * spec_loss + \
            self.multispectral * multispec_loss + self.commit * commit_loss

        with t.no_grad():
            sc = t.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn("l1", x_target, x_out, hps)
            linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            spectral_loss=spec_loss,
            multispectral_loss=multispec_loss,
            spectral_convergence=sc,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            linf_loss=linf_loss,
            commit_loss=commit_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics