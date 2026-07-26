import torch
from torch import nn
from src.rt_modules.common_model import CompressionModel
from src.rt_modules.layers.layers import SubpelConv2x, DepthConvBlock, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from src.rt_modules.layers.cuda_inference import round_and_to_int8

qp_shift = [0, 8, 4]

shuffle_ch = 3 * 8 * 8
g_ch_src_d = 32  #192
g_ch_recon = 320
g_ch_y = 128
g_ch_z = 128
g_ch_d = 256


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x):
        x1, ctx_t = self.forward_part1(x)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x):
        x1 = self.conv1(x)
        ctx_t = x1 
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        self.fuse_conv1_flag = False

    def forward(self, x, ctx):

        return self.forward_torch(x, ctx)

    def forward_torch(self, feature, ctx):
        feature = self.conv1(feature)
        feature = torch.cat((feature, ctx), dim=1)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        feature = self.down(feature)
        feature = torch.tanh(feature)        
        return feature


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx):

        return self.forward_torch(x, ctx)

    def forward_torch(self, x, ctx):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature 
        return feature



class ReconGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, 320, 1)

    def forward(self, x):
        return self.forward_torch(x)


    def forward_torch(self, x):
        out = self.conv(x)
        out = self.head(out)
        return out


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)



class RefFrame():
    def __init__(self):
        self.aux_feature = None
        self.feature = None
        self.last_recon = None
        self.refresh = None
        self.poc = None



class DMC(CompressionModel):
    def __init__(self, extra_qp=8):
        super().__init__(z_channel=g_ch_z, extra_qp=extra_qp)
        self.qp_shift = qp_shift

        self.feature_adaptor_i =  nn.Sequential(
            DepthConvBlock(32, g_ch_d),
        )
        self.feature_adaptor_refresh =  nn.Sequential(
            DepthConvBlock(32, g_ch_d),
        )
        self.feature_adaptor_aux= nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.feature_extractor = FeatureExtractor()
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)

        self.encoder = Encoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior = SpatialPrior()
        self.decoder = Decoder()
        self.recon_generation_net = ReconGeneration()
        self.recon_generation_aux =  nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            nn.Conv2d(g_ch_recon, 32, 1),
        )
        self.aux_extractor = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.dpb = []
        self.max_dpb_size = 1
        self.curr_poc = 0

    def reset_ref_feature(self):
        if len(self.dpb) > 0:
            self.dpb[0].feature = None

    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)
    def add_ref_frame(self, feature=None, aux_feature=None, last_recon = None , refresh = None, increase_poc=True):
        ref_frame = RefFrame()
        ref_frame.poc = self.curr_poc
        ref_frame.last_recon = last_recon
        ref_frame.aux_feature = aux_feature
        ref_frame.feature = feature
        ref_frame.refresh = refresh
        if len(self.dpb) >= self.max_dpb_size:
            self.dpb.pop(-1)
        self.dpb.insert(0, ref_frame)
        if increase_poc:
            self.curr_poc += 1

    def clear_dpb(self):
        self.dpb.clear()

    def set_curr_poc(self, poc):
        self.curr_poc = poc

    def apply_feature_adaptor(self):
        if self.dpb[0].feature is None and self.dpb[0].last_recon is not None:
            feature = self.feature_adaptor_i(self.dpb[0].last_recon)
            return feature
        
        feature = self.feature_adaptor_p(self.dpb[0].feature)
        return feature

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx):
        feature = self.decoder(y_hat, ctx)
        x_hat = self.recon_generation_net(feature)
        return x_hat, feature

    def prepare_feature_adaptor_i(self):
        if self.dpb[0].feature is None:
            self.dpb[0].last_recon = self.recon_generation_net(self.dpb[0].feature)
            self.reset_ref_feature()
    def reset_ref_features(self):
        if len(self.dpb) > 0:
            self.dpb[0].feature = None
            self.dpb[0].aux_feature = None

    def compress(self, x):
        device = x.device
        feature = self.apply_feature_adaptor()
        ctx, ctx_t = self.feature_extractor(feature)
        y = self.encoder(x, ctx)
        hyper_inp = self.pad_for_y(y)
        z = self.hyper_encoder(hyper_inp)

        z_hat, z_hat_write = round_and_to_int8(z)


        cuda_event_z_ready = torch.cuda.Event()
        cuda_event_z_ready.record()
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat = \
            self.compress_prior_2x(y, params, self.y_spatial_prior, write=True)

        cuda_event_y_ready = torch.cuda.Event()
        cuda_event_y_ready.record()
        feature = self.decoder(y_hat, ctx)
        x_hat = self.recon_generation_net(feature)
        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            self.entropy_coder.reset()
            cuda_event_z_ready.wait()
            self.bit_estimator_z.encode_z(z_hat_write, 0)
            cuda_event_y_ready.wait()
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        self.add_ref_frame(feature, feature)
        return x_hat, bit_stream


    def decompress(self, bit_stream, sps):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 256)
        self.bit_estimator_z.decode_z(z_size, 0)

        feature = self.apply_feature_adaptor()
        c1, ctx_t = self.feature_extractor.forward_part1(feature)

        z_hat = self.bit_estimator_z.get_z(z_size, device, dtype)
        params = self.res_prior_param_decoder(z_hat, ctx_t)

        ctx = self.feature_extractor.forward_part2(c1)
        infos = self.decompress_prior_2x_part1(params)
        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            y_hat = self.decompress_prior_2x_part2(params, self.y_spatial_prior, infos)
            cuda_event = torch.cuda.Event()
            cuda_event.record()
        cuda_event.wait()

        x_hat, feature = self.get_recon_and_feature(y_hat, ctx)
        self.add_ref_frame(feature, feature)
        return x_hat


    def forward(self, x):
        feature = self.apply_feature_adaptor()

        ctx, ctx_t = self.feature_extractor(feature)
        y = self.encoder(x, ctx)

        hyper_inp = self.pad_for_y(y)
        z = self.hyper_encoder(hyper_inp)

        z_hat = torch.clamp((self.quantize(z, "ste")), -128., 127.)

        params = self.res_prior_param_decoder(z_hat, ctx_t)

        y_q, scales_hat, y_hat = \
            self.compress_prior_2x(y, params, self.y_spatial_prior, write=False)
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx)
        self.add_ref_frame(feature, feature)

        _, _, H, W = x.size()
        pixel_num = H * W * 32 * 32

        bits_y = self.get_y_gaussian_bits(y_q, scales_hat)

        qp_tensor = self.get_q_tensor(0, device=x.device)

        bits_z = self.get_z_bits(z_hat, self.bit_estimator_z, qp_tensor)
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num

        return x_hat,feature, bpp_y, bpp_z


    def forward_dp(self, x, qp, fa_idx):
        qp_tensor = self.shift_qp(qp, fa_idx)
        return self.forward(x,qp_tensor)
        
    def shift_qp(self, qp, fa_idx):
        return qp + self.qp_shift[fa_idx]
