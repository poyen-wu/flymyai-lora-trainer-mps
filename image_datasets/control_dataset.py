import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0

def is_multiple_of_32(w, h):
    return (w % 32 == 0) and (h % 32 == 0)

def floor_to_multiple_of_32(w, h):
    new_w = max(32, (w // 32) * 32)
    new_h = max(32, (h // 32) * 32)
    return new_w, new_h

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {"16:9": (16, 9), "4:3": (4, 3), "1:1": (1, 1)}
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h
    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    return image.crop(crop_box)


class CustomImageDataset(Dataset):
    def __init__(
        self,
        img_dir,
        img_size=512,  # unused in this keep-original-size flow
        caption_type='txt',
        random_ratio=False,
        caption_dropout_rate=0.1,
        cached_text_embeddings=None,
        cached_image_embeddings=None,
        control_dir=None,
        cached_image_embeddings_control=None,
        skip_mismatched=True,  # skip pairs with different sizes instead of forcing resize
    ):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.lower().endswith(('.jpg', '.png', '.jpeg', '.webp', '.bmp'))]
        self.images.sort()
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.control_dir = control_dir
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.cached_control_image_embeddings = cached_image_embeddings_control
        self.skip_mismatched = skip_mismatched
        print('cached_text_embeddings', type(cached_text_embeddings))

    def __len__(self):
        return 999999

    def _load_and_fix(self, path, ratio_choice=None):
        img = Image.open(path).convert('RGB')
        w, h = img.size

        # only crop if the image is NOT already multiple-of-32
        if self.random_ratio and not is_multiple_of_32(w, h):
            if ratio_choice is None:
                ratio_choice = random.choice(["16:9", "default", "1:1", "4:3"])
            if ratio_choice != "default":
                img = crop_to_aspect_ratio(img, ratio_choice)
                w, h = img.size

        # only resize if not multiple-of-32
        if not is_multiple_of_32(w, h):
            new_w, new_h = floor_to_multiple_of_32(w, h)
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        tensor = torch.from_numpy((np.array(img, dtype=np.float32) / 127.5) - 1.0).permute(2, 0, 1)
        return tensor, img.size  # CHW tensor, (w,h)

    def __getitem__(self, _):
        for _attempt in range(10):
            try:
                idx = random.randint(0, len(self.images) - 1)
                img_path = self.images[idx]
                base = os.path.basename(img_path)

                # main image
                if self.cached_image_embeddings is None:
                    img_tensor, (iw, ih) = self._load_and_fix(img_path)
                else:
                    img_tensor = self.cached_image_embeddings[base]
                    iw, ih = None, None  # unknown here, but shapes must still align with control latents

                # control image
                if self.cached_control_image_embeddings is None:
                    if self.control_dir is None:
                        raise RuntimeError("control_dir is None but control image is required")
                    ctrl_path = os.path.join(self.control_dir, base)
                    control_tensor, (cw, ch) = self._load_and_fix(ctrl_path)
                else:
                    control_tensor = self.cached_control_image_embeddings[base]
                    cw, ch = None, None

                # Ensure main and control sizes match when using raw images (not precomputed latents)
                if self.cached_image_embeddings is None and self.cached_control_image_embeddings is None:
                    if (iw, ih) != (cw, ch):
                        if self.skip_mismatched:
                            # try another sample
                            continue
                        else:
                            # if you really want to force-match, uncomment next line
                            # control_tensor = torch.nn.functional.interpolate(control_tensor.unsqueeze(0), size=(img_tensor.shape[-2], img_tensor.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)
                            pass

                # captions / text embeddings
                txt_path = os.path.splitext(img_path)[0] + f'.{self.caption_type}'
                if self.cached_text_embeddings is None:
                    prompt = open(txt_path, encoding='utf-8').read()
                    if throw_one(self.caption_dropout_rate):
                        return img_tensor, " ", control_tensor
                    else:
                        return img_tensor, prompt, control_tensor
                else:
                    txt = os.path.basename(txt_path)
                    key = txt + 'empty_embedding' if throw_one(self.caption_dropout_rate) else txt
                    pe = self.cached_text_embeddings[key]
                    return img_tensor, pe['prompt_embeds'], pe['prompt_embeds_mask'], control_tensor

            except Exception as e:
                print(f"[Dataset] Failed to load sample (attempt retry): {e}")
                continue

        raise RuntimeError("Failed to load a valid sample after several retries")


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)