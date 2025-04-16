import cv2
import numpy as np
from torchvision.transforms import Compose

class Resize:
    def __init__(
        self,
        width,
        height,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    ):
        self.target_width = width
        self.target_height = height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.image_interpolation_method = image_interpolation_method

    def __call__(self, sample):
        image = sample["image"]
        height, width = image.shape[:2]

        if self.keep_aspect_ratio:
            if self.resize_method == "lower_bound":
                scale = max(self.target_width / width, self.target_height / height)
            else:
                raise ValueError("Unknown resize method")

            new_width = int(round(width * scale))
            new_height = int(round(height * scale))

            if self.ensure_multiple_of is not None:
                new_width -= new_width % self.ensure_multiple_of
                new_height -= new_height % self.ensure_multiple_of
        else:
            new_width = self.target_width
            new_height = self.target_height

        image = cv2.resize(image, (new_width, new_height), interpolation=self.image_interpolation_method)
        sample["image"] = image
        return sample

class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image = sample["image"].astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        sample["image"] = image
        return sample

class PrepareForNet:
    def __call__(self, sample):
        image = sample["image"]
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        sample["image"] = torch.from_numpy(image)
        return sample

small_transform = Compose([
    Resize(256, 256, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet()
])
