from albumentations import (HorizontalFlip, ShiftScaleRotate, RandomContrast, RandomBrightness,RandomBrightnessContrast, Compose)

#  图像增强
def get_augmentations(augmentation, p):
    if augmentation == 'valid':
        augmentations = Compose([
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast( brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7)
        ], p=p)

    else:
        raise ValueError("Unknown Augmentations")

    return augmentations
