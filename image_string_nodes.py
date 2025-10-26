import torch

class ImageStringBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "string_1": ("STRING", {"default": '', "forceInput": True}),
            },
            "optional": {
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
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "strings")
    FUNCTION = "build"
    CATEGORY = "lianlaoshi/image"
    DESCRIPTION = (
        "Creates a batch of images paired with strings.\n"
        "Predeclares up to 10 pairs of inputs; uses connected pairs automatically."
    )

    def build(self, image_1, string_1, **kwargs):
        # Initialize batch with the first image and strings list
        images = image_1
        strings = []
        if isinstance(string_1, list):
            strings.append(string_1[0] if string_1 else "")
        else:
            strings.append(string_1)

        first_image_shape = image_1.shape
        device = image_1.device
        dtype = image_1.dtype

        # Determine the highest index of connected optional inputs
        last_idx = 1
        for i in range(2, 11):
            if (f"image_{i}" in kwargs and kwargs.get(f"image_{i}") is not None) or \
               (f"string_{i}" in kwargs and kwargs.get(f"string_{i}") is not None):
                last_idx = i

        for i in range(2, last_idx + 1):
            new_image = kwargs.get(f"image_{i}")
            new_string = kwargs.get(f"string_{i}", "")

            if new_image is None:
                # If an image input is missing, pad with a black image of the first image's shape
                new_image = torch.zeros(first_image_shape, device=device, dtype=dtype)

            # Concatenate along batch dimension
            images = torch.cat((images, new_image), dim=0)

            # Append string (empty string if not provided)
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


NODE_CLASS_MAPPINGS = {
    "ImageStringBatch": ImageStringBatch,
    "SelectImageString": SelectImageString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageStringBatch": "lian 图像字符串批次构建",
    "SelectImageString": "lian 图像字符串选择",
}