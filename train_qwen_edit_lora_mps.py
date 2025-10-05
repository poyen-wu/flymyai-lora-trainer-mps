import argparse
import copy
import logging
import os
import shutil
import gc
from contextlib import nullcontext

import torch
from tqdm.auto import tqdm

import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import QwenImageEditPipeline

from image_datasets.control_dataset import loader
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from PIL import Image
import numpy as np

# Optional deps
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except Exception:
    bnb = None
    BNB_AVAILABLE = False

try:
    from optimum.quanto import quantize, qfloat8, freeze
    QUANTO_AVAILABLE = True
except Exception:
    quantize = None
    qfloat8 = None
    freeze = None
    QUANTO_AVAILABLE = False

# Device flags/helpers
IS_CUDA = torch.cuda.is_available()
IS_MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

def empty_cache():
    if IS_CUDA:
        torch.cuda.empty_cache()
    elif IS_MPS:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Training script without accelerate.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config (OmegaConf YAML)",
    )
    args = parser.parse_args()
    return args.config

# -------------------------
# New: dataset scanning + size helpers
# -------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def list_images(dir_path):
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(IMG_EXTS)])

def is_multiple_of_32(w, h):
    return (w % 32 == 0) and (h % 32 == 0)

def floor_to_multiple_of_32(w, h):
    new_w = max(32, (w // 32) * 32)
    new_h = max(32, (h // 32) * 32)
    return new_w, new_h

def resize_to_multiple_of_32(pil_img):
    w, h = pil_img.size
    if is_multiple_of_32(w, h):
        return pil_img, (w, h), False
    new_w, new_h = floor_to_multiple_of_32(w, h)
    if (new_w, new_h) != (w, h):
        pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        return pil_img, (new_w, new_h), True
    return pil_img, (w, h), False

def scan_dataset_and_warn(img_dir, control_dir=None, logger=None):
    dims_count_img = {}
    dims_count_ctrl = {}
    invalid_imgs = []
    invalid_ctrls = []
    pair_mismatches = []

    # Map base name -> (W,H) for pair checking
    base_to_img_dims = {}
    base_to_ctrl_dims = {}

    # Scan img_dir
    img_files = list_images(img_dir)
    for fname in img_files:
        p = os.path.join(img_dir, fname)
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
        except Exception as e:
            if logger:
                logger.warning(f"Failed to read image: {p} ({e})")
            continue

        dims_count_img[(w, h)] = dims_count_img.get((w, h), 0) + 1
        base = os.path.splitext(fname)[0]
        base_to_img_dims[base] = (w, h)
        if not is_multiple_of_32(w, h):
            invalid_imgs.append((fname, (w, h)))

    # Scan control_dir
    if control_dir is not None:
        ctrl_files = list_images(control_dir)
        for fname in ctrl_files:
            p = os.path.join(control_dir, fname)
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    w, h = im.size
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to read control image: {p} ({e})")
                continue

            dims_count_ctrl[(w, h)] = dims_count_ctrl.get((w, h), 0) + 1
            base = os.path.splitext(fname)[0]
            base_to_ctrl_dims[base] = (w, h)
            if not is_multiple_of_32(w, h):
                invalid_ctrls.append((fname, (w, h)))

        # Pair mismatch report
        shared_bases = set(base_to_img_dims.keys()) & set(base_to_ctrl_dims.keys())
        for base in sorted(shared_bases):
            if base_to_img_dims[base] != base_to_ctrl_dims[base]:
                pair_mismatches.append((base, base_to_img_dims[base], base_to_ctrl_dims[base]))

    # Logs
    if logger:
        if dims_count_img:
            logger.info("Image dir dimension combos (W,H -> count):")
            for (w, h), c in sorted(dims_count_img.items()):
                logger.info(f"  {w}x{h}: {c}")

        if dims_count_ctrl:
            logger.info("Control dir dimension combos (W,H -> count):")
            for (w, h), c in sorted(dims_count_ctrl.items()):
                logger.info(f"  {w}x{h}: {c}")

        if invalid_imgs:
            logger.warning("Images not multiple-of-32 in img_dir:")
            for name, (w, h) in invalid_imgs:
                logger.warning(f"  {name}: {w}x{h}")

        if invalid_ctrls:
            logger.warning("Images not multiple-of-32 in control_dir:")
            for name, (w, h) in invalid_ctrls:
                logger.warning(f"  {name}: {w}x{h}")

        if pair_mismatches:
            logger.warning("Mismatched sizes between paired images:")
            for base, idims, cdims in pair_mismatches:
                logger.warning(f"  {base}: img={idims}, control={cdims}")

    return {
        "dims_count_img": dims_count_img,
        "dims_count_ctrl": dims_count_ctrl,
        "invalid_imgs": invalid_imgs,
        "invalid_ctrls": invalid_ctrls,
        "pair_mismatches": pair_mismatches,
    }

# -------------------------
# AMP / device helpers
# -------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"

def setup_amp(mixed_precision: str, device_type: str):
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if mixed_precision in ("fp16", "bf16"):
        if device_type == "cuda":
            autocast_cm = torch.cuda.amp.autocast(dtype=dtype)
        else:
            autocast_cm = torch.autocast(device_type=device_type, dtype=dtype)
    else:
        autocast_cm = nullcontext()

    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and mixed_precision == "fp16"))
    return dtype, autocast_cm, scaler

# -------------------------

def main():
    # Load config
    args = OmegaConf.load(parse_args())

    # Defaults from original script (can be overridden in your config if desired)
    args.save_cache_on_disk = args.get("save_cache_on_disk", False)
    args.precompute_text_embeddings = args.get("precompute_text_embeddings", True)
    args.precompute_image_embeddings = args.get("precompute_image_embeddings", True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Device + AMP
    device, device_type = get_device()
    mixed_precision = args.get("mixed_precision", "no")
    weight_dtype, amp_autocast, scaler = setup_amp(mixed_precision, device_type)

    # Disable CUDA-only features when not on CUDA
    if args.get("quantize", False) and not (IS_CUDA and QUANTO_AVAILABLE):
        logger.info("Disabling quantize: CUDA or optimum.quanto not available on this platform.")
        args.quantize = False
    if args.get("adam8bit", False) and not (IS_CUDA and BNB_AVAILABLE):
        logger.info("Falling back to AdamW: bitsandbytes not available or CUDA not present.")
        args.adam8bit = False

    # Ensure output dir exists
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        os.makedirs(logging_dir, exist_ok=True)

    # Scan dataset sizes and warn if needed
    logger.info("Scanning dataset for dimension combos and invalid sizes (not multiple-of-32)...")
    _ = scan_dataset_and_warn(args.data_config.img_dir, args.data_config.control_dir, logger=logger)

    # Text encoding pipeline (for precompute)
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(device)

    cached_text_embeddings = None
    txt_cache_dir = None
    cached_image_embeddings = None
    img_cache_dir = None
    cached_image_embeddings_control = None
    img_cache_dir_control = None

    # Cache base dir
    cache_dir = None
    if args.precompute_text_embeddings or args.precompute_image_embeddings:
        cache_dir = os.path.join(args.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

    # Precompute text embeddings
    if args.precompute_text_embeddings:
        with torch.no_grad():
            if args.save_cache_on_disk:
                txt_cache_dir = os.path.join(cache_dir, "text_embs")
                os.makedirs(txt_cache_dir, exist_ok=True)
            else:
                cached_text_embeddings = {}

            for img_name in tqdm([i for i in os.listdir(args.data_config.control_dir) if i.lower().endswith(IMG_EXTS)]):
                img_path = os.path.join(args.data_config.control_dir, img_name)
                txt_key = img_name.split('.')[0] + '.txt'
                txt_path = os.path.join(args.data_config.img_dir, txt_key)

                img = Image.open(img_path).convert('RGB')
                img, (w_used, h_used), _ = resize_to_multiple_of_32(img)

                prompt = open(txt_path, encoding='utf-8').read()
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=img,
                    prompt=[prompt],
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                if args.save_cache_on_disk:
                    torch.save(
                        {
                            'prompt_embeds': prompt_embeds[0].to('cpu'),
                            'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu'),
                        },
                        os.path.join(txt_cache_dir, txt_key + '.pt'),
                    )
                else:
                    cached_text_embeddings[txt_key] = {
                        'prompt_embeds': prompt_embeds[0].to('cpu'),
                        'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu'),
                    }

                # empty embedding (per-sample)
                prompt_embeds_empty, prompt_embeds_mask_empty = text_encoding_pipeline.encode_prompt(
                    image=img,
                    prompt=[' '],
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                if args.save_cache_on_disk:
                    torch.save(
                        {
                            'prompt_embeds': prompt_embeds_empty[0].to('cpu'),
                            'prompt_embeds_mask': prompt_embeds_mask_empty[0].to('cpu'),
                        },
                        os.path.join(txt_cache_dir, txt_key + '_empty_embedding.pt'),
                    )
                else:
                    cached_text_embeddings[txt_key + 'empty_embedding'] = {
                        'prompt_embeds': prompt_embeds_empty[0].to('cpu'),
                        'prompt_embeds_mask': prompt_embeds_mask_empty[0].to('cpu'),
                    }

    # VAE for image embeddings
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(device, dtype=weight_dtype)

    # Precompute image latents
    if args.precompute_image_embeddings:
        if args.save_cache_on_disk:
            img_cache_dir = os.path.join(cache_dir, "img_embs")
            os.makedirs(img_cache_dir, exist_ok=True)
        else:
            cached_image_embeddings = {}

        with torch.no_grad():
            for img_name in tqdm([i for i in os.listdir(args.data_config.img_dir) if i.lower().endswith(IMG_EXTS)]):
                img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                img, (w_used, h_used), _ = resize_to_multiple_of_32(img)

                img_np = (np.array(img) / 127.5) - 1.0
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # B,C,H,W
                pixel_values = img_t.unsqueeze(2).to(dtype=weight_dtype, device=device)  # B,C,F,H,W

                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if args.save_cache_on_disk:
                    torch.save(pixel_latents, os.path.join(img_cache_dir, img_name + '.pt'))
                    del pixel_latents
                else:
                    cached_image_embeddings[img_name] = pixel_latents

        if args.save_cache_on_disk:
            img_cache_dir_control = os.path.join(cache_dir, "img_embs_control")
            os.makedirs(img_cache_dir_control, exist_ok=True)
        else:
            cached_image_embeddings_control = {}

        with torch.no_grad():
            for img_name in tqdm([i for i in os.listdir(args.data_config.control_dir) if i.lower().endswith(IMG_EXTS)]):
                img = Image.open(os.path.join(args.data_config.control_dir, img_name)).convert('RGB')
                img, (w_used, h_used), _ = resize_to_multiple_of_32(img)

                img_np = (np.array(img) / 127.5) - 1.0
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
                pixel_values = img_t.unsqueeze(2).to(dtype=weight_dtype, device=device)

                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                if args.save_cache_on_disk:
                    torch.save(pixel_latents, os.path.join(img_cache_dir_control, img_name + '.pt'))
                    del pixel_latents
                else:
                    cached_image_embeddings_control[img_name] = pixel_latents

        vae.to('cpu')
        empty_cache()
        text_encoding_pipeline.to("cpu")
        empty_cache()

    # Free text_encoding_pipeline if not needed further
    del text_encoding_pipeline
    gc.collect()

    # Transformer
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )

    # Optional quantization (CUDA only)
    if args.get("quantize", False) and QUANTO_AVAILABLE and IS_CUDA:
        torch_dtype = weight_dtype
        # Quantize block by block to keep memory in check
        all_blocks = list(flux_transformer.transformer_blocks)
        for block in tqdm(all_blocks, desc="Quantizing blocks"):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
        flux_transformer.to(device, dtype=torch_dtype)
        quantize(flux_transformer, weights=qfloat8)
        freeze(flux_transformer)

    # LoRA config
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Move transformer to device/dtype
    if args.get("quantize", False) and QUANTO_AVAILABLE and IS_CUDA:
        flux_transformer.to(device)
    else:
        flux_transformer.to(device, dtype=weight_dtype)

    # Add LoRA adapters and enable grads only for LoRA params
    flux_transformer.add_adapter(lora_config)
    flux_transformer.requires_grad_(False)
    for n, param in flux_transformer.named_parameters():
        if 'lora' in n:
            param.requires_grad = True
    flux_transformer.train()
    flux_transformer.enable_gradient_checkpointing()

    # Optimizer
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    if args.get("adam8bit", False) and BNB_AVAILABLE and IS_CUDA:
        optimizer = bnb.optim.Adam8bit(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    else:
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataloader from your custom loader
    train_dataloader = loader(
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings,
        cached_image_embeddings_control=cached_image_embeddings_control,
        **args.data_config
    )

    # LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # VAE scale factor from config
    td = getattr(vae, "temperal_downsample", None)
    if td is None:
        td = getattr(vae, "temporal_downsample", None)
    vae_scale_factor = 2 ** len(td) if td is not None else 8

    # Helper to get sigmas for timesteps
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Training
    global_step = 0
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    logging.getLogger(__name__).info("***** Running training *****")
    logging.getLogger(__name__).info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logging.getLogger(__name__).info(f"  Total train batch size (accumulation) = {total_batch_size}")
    logging.getLogger(__name__).info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    progress_bar = tqdm(range(0, args.max_train_steps), desc="Steps")
    grad_accum = max(1, args.gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)

    train_loss_accum = 0.0

    # Pre-calc shapes may vary per sample, so keep dynamic parts inside loop
    for epoch in range(1):
        for step, batch in enumerate(train_dataloader):
            if args.precompute_text_embeddings:
                img, prompt_embeds, prompt_embeds_mask, control_img = batch
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)
                prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32, device=device)
                control_img = control_img.to(dtype=weight_dtype, device=device)
            else:
                img, prompts = batch

            with torch.no_grad():
                if not args.precompute_image_embeddings:
                    pixel_values = img.to(dtype=weight_dtype, device=device)
                    pixel_values = pixel_values.unsqueeze(2)  # B,C,F,H,W
                    pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                else:
                    pixel_latents = img.to(dtype=weight_dtype, device=device)

                pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                control_img = control_img.permute(0, 2, 1, 3, 4)

                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, 1, vae.config.z_dim, 1, 1)
                    .to(pixel_latents.device, pixel_latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                    pixel_latents.device, pixel_latents.dtype
                )
                pixel_latents = (pixel_latents - latents_mean) * latents_std
                control_img = (control_img - latents_mean) * latents_std

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

            sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                noisy_model_input,
                bsz,
                noisy_model_input.shape[2],
                noisy_model_input.shape[3],
                noisy_model_input.shape[4],
            )
            packed_control_img = QwenImageEditPipeline._pack_latents(
                control_img,
                bsz,
                control_img.shape[2],
                control_img.shape[3],
                control_img.shape[4],
            )
            img_shapes = [[
                (1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                (1, control_img.shape[3] // 2, control_img.shape[4] // 2)
            ]] * bsz

            packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)

            with torch.no_grad():
                if not args.precompute_text_embeddings:
                    # If you ever set precompute_text_embeddings=False in config, ensure you create the pipeline earlier
                    raise RuntimeError("precompute_text_embeddings is False but text pipeline was deleted earlier.")
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

            # forward + loss under autocast
            with amp_autocast:
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)

                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                per_sample = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = per_sample.mean()

            # gradient accumulation
            loss_for_backward = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            train_loss_accum += loss.item()

            do_step = ((step + 1) % grad_accum == 0)
            if do_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(step_loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logging.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logging.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    # Save LoRA weights
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(flux_transformer)
                    )
                    QwenImagePipeline.save_lora_weights(
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,
                    )
                    logging.info(f"Saved state to {save_path}")

                train_loss_accum = 0.0

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    # Done
    logging.info("Training completed.")


if __name__ == "__main__":
    main()
