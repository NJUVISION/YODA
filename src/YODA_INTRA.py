import torch
import torch.nn.functional as F
from diffusers import AutoencoderDC, SanaTransformer2DModel, SCMScheduler
from peft import LoraConfig

from src.Codec_Intra import DMCI
from utils.utils import get_state_dict

def load_matching_weights(module, weights):
    model_weights = module.state_dict()
    weights = {
        name: value
        for name, value in weights.items()
        if name in model_weights and value.shape == model_weights[name].shape
    }
    module.load_state_dict(weights, strict=False)
    return len(weights)


def sched_one_step(sched, timesteps, latent_hat, model_pred):
    alpha_t = sched.alphas_cumprod[timesteps]  # shape: scalar
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    x0 = (latent_hat - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t
    return x0

class Yoda_Intra(torch.nn.Module):
    def __init__(self, sd_path=None, args=None):
        super().__init__()

        print("[SD-Turbo]: Building SD-Turbo ......")
        self.sched = SCMScheduler.from_pretrained(
        sd_path, 
        subfolder="scheduler"
    )
        self.guidance_scale = 1.07
        config = AutoencoderDC.load_config(sd_path, subfolder="vae")
        vae = AutoencoderDC.from_config(config)
        transformer = SanaTransformer2DModel.from_pretrained(sd_path, subfolder="transformer")
        self.transformer, self.vae = transformer, vae
        self.timesteps = torch.tensor([args.timestep_i], device="cuda", dtype=torch.long)  
        self.prompt_embeds = torch.load(args.prompt_embeds_path, map_location="cpu")
        self.prompt_attention_mask = torch.load(
            args.prompt_attention_mask_path, map_location="cpu"
        )

        target_modules_transformer = [
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "proj_in", "proj_out"   
        ]

        lora_rank_transformer = args.lora_rank_transformer

        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)


        transformer_lora_config = LoraConfig(
            r=lora_rank_transformer,
            init_lora_weights="gaussian",
            target_modules=target_modules_transformer,
        )
        self.transformer.add_adapter(transformer_lora_config)

        self.codec = DMCI()

        if args.pretrained_i_weights:
            checkpoint = get_state_dict(args.pretrained_i_weights)

            loaded = {
                "codec": load_matching_weights(
                    self.codec, checkpoint["state_dict_codec"]
                ),
                "vae": load_matching_weights(
                    self.vae, checkpoint["state_dict_vae"]
                ),
                "transformer": load_matching_weights(
                    self.transformer, checkpoint["state_dict_transformer"]
                ),
            }
            print(f"Loaded checkpoint: {loaded}")


    def forward(self, c_t):
        lq_latent = self.vae.encode(c_t).latent * self.vae.config.scaling_factor
        lq_latent_hat , bpp_y, bpp_z = self.codec(lq_latent) 

        batch_size = lq_latent.shape[0]

        if self.prompt_embeds.shape[0] != batch_size:
            prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
            prompt_attention_mask = self.prompt_attention_mask.expand(batch_size, -1)
        else:
            prompt_embeds = self.prompt_embeds
            prompt_attention_mask = self.prompt_attention_mask
        guidance = torch.full([1], 0.0, device=lq_latent_hat.device, dtype=torch.float32)
        guidance = guidance.expand(lq_latent_hat.shape[0]).to(lq_latent_hat.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale



        latents = lq_latent_hat * self.sched.config.sigma_data
        latents_model_input = latents / self.sched.config.sigma_data
        self.sched.set_timesteps(1, device=latents.device, intermediate_timesteps=None)
        timesteps = self.sched.timesteps
        timesteps = timesteps[:-1]

        if hasattr(self.sched, "set_begin_index"):
            self.sched.set_begin_index(0)

        timestep = timesteps.expand(latents.shape[0])
        scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
        scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)
        latent_model_input = latents_model_input * torch.sqrt(
            scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2
        )
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds.to(
                device=latent_model_input.device,
                dtype=self.transformer.dtype,
            ),
            encoder_attention_mask=prompt_attention_mask.to(
                device=latent_model_input.device
            ),
            timestep=scm_timestep,
            guidance=guidance,
            return_dict=False,
        )[0]    
        noise_pred = (
            (1 - 2 * scm_timestep_expanded) * latent_model_input
            + (1 - 2 * scm_timestep_expanded + 2 * scm_timestep_expanded**2) * noise_pred
        ) / torch.sqrt(scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2)    
        noise_pred = noise_pred.float() * self.sched.config.sigma_data

        extra_step_kwargs = {}
        latents, denoised = self.sched.step(
            noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
        )
        latents = denoised / self.sched.config.sigma_data

        x_denoised = latents 
        mse_latent = F.mse_loss(x_denoised, lq_latent)

        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
        return{"output_image": output_image, "bpp_y": bpp_y , "bpp_z": bpp_z , "mse_latent": mse_latent, 'x_denoised': lq_latent_hat}

    def compress(self, c_t):
        lq_latent = self.vae.encode(c_t).latent * self.vae.config.scaling_factor
        lq_latent_hat, bit_stream = self.codec.compress(lq_latent)

        batch_size = lq_latent.shape[0]

        if self.prompt_embeds.shape[0] != batch_size:
            prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
            prompt_attention_mask = self.prompt_attention_mask.expand(batch_size, -1)
        else:
            prompt_embeds = self.prompt_embeds
            prompt_attention_mask = self.prompt_attention_mask
        guidance = torch.full([1], 0.0, device=lq_latent_hat.device, dtype=torch.float32)
        guidance = guidance.expand(lq_latent_hat.shape[0]).to(lq_latent_hat.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale



        latents = lq_latent_hat * self.sched.config.sigma_data
        latents_model_input = latents / self.sched.config.sigma_data
        self.sched.set_timesteps(1, device=latents.device, intermediate_timesteps=None)
        timesteps = self.sched.timesteps
        timesteps = timesteps[:-1]

        if hasattr(self.sched, "set_begin_index"):
            self.sched.set_begin_index(0)

        timestep = timesteps.expand(latents.shape[0])
        scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
        scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)
        latent_model_input = latents_model_input * torch.sqrt(
            scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2
        )
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds.to(
                device=latent_model_input.device,
                dtype=self.transformer.dtype,
            ),
            encoder_attention_mask=prompt_attention_mask.to(
                device=latent_model_input.device
            ),
            timestep=scm_timestep,
            guidance=guidance,
            return_dict=False,
        )[0]  
        noise_pred = (
            (1 - 2 * scm_timestep_expanded) * latent_model_input
            + (1 - 2 * scm_timestep_expanded + 2 * scm_timestep_expanded**2) * noise_pred
        ) / torch.sqrt(scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2)    
        noise_pred = noise_pred.float() * self.sched.config.sigma_data

        extra_step_kwargs = {}
        latents, denoised = self.sched.step(
            noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
        )
        latents = denoised / self.sched.config.sigma_data

        x_denoised = latents 
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0]

        result = {
            "bit_stream": bit_stream,
            'x_denoised': x_denoised,
            'x_hat': output_image
        }
        return result
    
    def decompress(self,  bit_stream, sps):

        lq_latent_hat = self.codec.decompress(bit_stream, sps)
        batch_size = lq_latent_hat.shape[0]

        if self.prompt_embeds.shape[0] != batch_size:
            prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
            prompt_attention_mask = self.prompt_attention_mask.expand(batch_size, -1)
        else:
            prompt_embeds = self.prompt_embeds
            prompt_attention_mask = self.prompt_attention_mask
        guidance = torch.full([1], 0.0, device=lq_latent_hat.device, dtype=torch.float32)
        guidance = guidance.expand(lq_latent_hat.shape[0]).to(lq_latent_hat.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale



        latents = lq_latent_hat * self.sched.config.sigma_data
        latents_model_input = latents / self.sched.config.sigma_data
        self.sched.set_timesteps(1, device=latents.device, intermediate_timesteps=None)
        timesteps = self.sched.timesteps
        timesteps = timesteps[:-1]

        if hasattr(self.sched, "set_begin_index"):
            self.sched.set_begin_index(0)

        timestep = timesteps.expand(latents.shape[0])
        scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
        scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)
        latent_model_input = latents_model_input * torch.sqrt(
            scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2
        )
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds.to(
                device=latent_model_input.device,
                dtype=self.transformer.dtype,
            ),
            encoder_attention_mask=prompt_attention_mask.to(
                device=latent_model_input.device
            ),
            timestep=scm_timestep,
            guidance=guidance,
            return_dict=False,
        )[0]   
        noise_pred = (
            (1 - 2 * scm_timestep_expanded) * latent_model_input
            + (1 - 2 * scm_timestep_expanded + 2 * scm_timestep_expanded**2) * noise_pred
        ) / torch.sqrt(scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2)    
        noise_pred = noise_pred.float() * self.sched.config.sigma_data

        extra_step_kwargs = {}
        latents, denoised = self.sched.step(
            noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
        )
        latents = denoised / self.sched.config.sigma_data

        x_denoised = latents 
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor, return_dict=False)[0]
        # Decoder

        return {'x_hat': output_image, 'x_denoised': x_denoised}
    
