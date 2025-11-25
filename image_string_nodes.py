import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def _center_crop_to_aspect_nhwc(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    n, h, w, c = t.shape
    ar_t = target_w / float(target_h)
    ar_s = w / float(h)
    if abs(ar_s - ar_t) < 1e-6:
        return t
    if ar_s > ar_t:
        new_w = max(1, int(round(h * ar_t)))
        x0 = max(0, (w - new_w) // 2)
        return t[:, :, x0:x0 + new_w, :]
    else:
        new_h = max(1, int(round(w / ar_t)))
        y0 = max(0, (h - new_h) // 2)
        return t[:, y0:y0 + new_h, :, :]

def _resize_lanczos_nhwc(t: torch.Tensor, target_h: int, target_w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    n = t.shape[0]
    out_list = []
    for i in range(n):
        arr = (t[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        pil = pil.resize((target_w, target_h), resample=Image.LANCZOS)
        arr2 = np.asarray(pil).astype(np.float32) / 255.0
        out_list.append(torch.from_numpy(arr2))
    out = torch.stack(out_list, dim=0)
    return out.to(device=device, dtype=dtype)

class ImageStringBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "optional": {
                "string_1": ("STRING", {"default": '', "forceInput": True}),                
                "image_2": ("IMAGE",),
                "string_2": ("STRING", {"default": '', "forceInput": True}),
                "image_3": ("IMAGE",),
                "string_3": ("STRING", {"default": '', "forceInput": True}),
                "image_4": ("IMAGE",),
                "string_4": ("STRING", {"default": '', "forceInput": True}),
                "image_5": ("IMAGE",),
                "string_5": ("STRING", {"default": '', "forceInput": True}),
                "image_6": ("IMAGE",),
                "string_6": ("STRING", {"default": '', "forceInput": True}),
                "image_7": ("IMAGE",),
                "string_7": ("STRING", {"default": '', "forceInput": True}),
                "image_8": ("IMAGE",),
                "string_8": ("STRING", {"default": '', "forceInput": True}),
                "image_9": ("IMAGE",),
                "string_9": ("STRING", {"default": '', "forceInput": True}),
                "image_10": ("IMAGE",),
                "string_10": ("STRING", {"default": '', "forceInput": True}),
                "target_width": ("INT", {"default": 832, "min": 0, "max": 4096, "step": 1, "display": "number"}),
                "target_height": ("INT", {"default": 480, "min": 0, "max": 4096, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "strings")
    FUNCTION = "combine""build"
    CATEGORY = "lianlaoshi/image"
    DESCRIPTION = (
        "Creates a batch of images paired with strings.\n"
        "Predeclares up to 10 pairs of inputs; uses connected pairs automatically.\n"
        "Optionally resizes all images to the specified width/height to avoid size mismatches."
    )

    def build(self, image_1, string_1="", **kwargs):
        # Initialize batch with the first image and strings list
        images = image_1
        strings = []
        if isinstance(string_1, list):
            strings.append(string_1[0] if string_1 else "")
        else:
            strings.append(string_1)

        first_image_shape = image_1.shape  # [N, H, W, C]
        device = image_1.device
        dtype = image_1.dtype
        req_w = kwargs.get("target_width", 0)
        req_h = kwargs.get("target_height", 0)
        target_w = req_w if isinstance(req_w, int) and req_w > 0 else first_image_shape[2]
        target_h = req_h if isinstance(req_h, int) and req_h > 0 else first_image_shape[1]
        target_c = first_image_shape[3]

        # 先对首批图像进行中心裁剪与 Lanczos 缩放到目标尺寸
        if images.shape[1] != target_h or images.shape[2] != target_w:
            cropped = _center_crop_to_aspect_nhwc(images, target_h, target_w)
            images = _resize_lanczos_nhwc(cropped, target_h, target_w, dtype, device)

        # 遍历所有可选输入：仅当有图像时加入列表，缺失则跳过
        for i in range(2, 11):
            new_image = kwargs.get(f"image_{i}")
            if new_image is None:
                continue
            new_string = kwargs.get(f"string_{i}", "")

            # 对齐设备与dtype
            new_image = new_image.to(device=device, dtype=dtype)
            # 通道数必须一致
            assert new_image.shape[-1] == target_c, "通道数不一致，无法拼接"
            # 统一到目标尺寸：中心裁剪到目标纵横比，再用 Lanczos 缩放
            if new_image.shape[1] != target_h or new_image.shape[2] != target_w:
                cropped = _center_crop_to_aspect_nhwc(new_image, target_h, target_w)
                new_image = _resize_lanczos_nhwc(cropped, target_h, target_w, dtype, device)

            # Concatenate along batch dimension
            images = torch.cat((images, new_image), dim=0)

            # Append string (空字符串兜底)
            if isinstance(new_string, list):
                strings.append(new_string[0] if new_string else "")
            else:
                strings.append(new_string)

        return (images, strings)


class SelectImageString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strings": ("STRING", {"default": ''}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "string")
    FUNCTION = "select"
    CATEGORY = "lianlaoshi/image"
    DESCRIPTION = (
        "Selects a single image and string from the batch by index (0-based)."
    )

    def select(self, images, strings, index):
        # Determine valid indices for images and strings
        batch_size = images.shape[0]
        img_idx = max(0, min(index, max(0, batch_size - 1)))

        if isinstance(strings, list):
            str_len = len(strings)
            str_idx = max(0, min(index, max(0, str_len - 1)))
            selected_string = strings[str_idx] if str_len > 0 else ""
        else:
            selected_string = strings

        # Select image and keep batch dimension
        selected_image = images[img_idx].unsqueeze(0)
        return (selected_image, selected_string)

# 动态成对输入版本：支持根据 inputcount 动态添加 image_X 与 string_X
class ImageStringBatchMulti:
    @classmethod
    def INPUT_TYPES(cls):
        cls._inputcount = 4   # 默认至少4对
        return {
            "required": {
                "inputcount":   ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "label": "Pairs (1~20)"}),
                "target_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 64}),
                "target_height":("INT", {"default": 480, "min": 64, "max": 2048, "step": 64}),
            },
            "optional": {
                "image_1": ("IMAGE", {"forceInput": True}),
                "string_1": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "hidden": { },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "strings")
    FUNCTION = "combine"
    CATEGORY = "lianlaoshi/image"
    DESCRIPTION = (
        "批量接收圆点式图像与字符串列表，统一尺寸后输出。\n"
        "用 update_inputs 按钮刷新前端显示。"
    )

    def combine(self, inputcount, target_width, target_height, **kwargs):
        images = []
        strings = []
        device, dtype = None, None

        # 按顺序收集 image_i / string_i
        for i in range(1, inputcount + 1):
            img = kwargs.get(f"image_{i}")
            txt = kwargs.get(f"string_{i}", "")
            if img is not None:
                if device is None:
                    device, dtype = img.device, img.dtype
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                # 统一尺寸
                B, H, W, C = img.shape
                if H != target_height or W != target_width:
                    img = _resize_lanczos_nhwc(
                        _center_crop_to_aspect_nhwc(img, target_height, target_width),
                        target_height, target_width, dtype, device
                    )
                images.append(img)
                # 字符串取第一个非空或空串
                if isinstance(txt, list):
                    txt = txt[0] if txt else ""
                strings.append(str(txt))

        # 缺图时给一张零图兜底
        if not images:
            images.append(torch.zeros((1, target_height, target_width, 3)))
            strings.append("")

        # 合并 batch
        image_batch = torch.cat(images, dim=0)
        # 补齐字符串数量
        if len(strings) < image_batch.shape[0]:
            strings += [""] * (image_batch.shape[0] - len(strings))

        return (image_batch, strings)


NODE_CLASS_MAPPINGS = {
    "ImageStringBatch": ImageStringBatch,
    "SelectImageString": SelectImageString,
    "ImageStringBatchMulti": ImageStringBatchMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageStringBatch": "lian 图像字符串批次构建",
    "SelectImageString": "lian 图像字符串选择",
    "ImageStringBatchMulti": "lian 图像字符串批次构建·动态",
}
