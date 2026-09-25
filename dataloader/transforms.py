import albumentations as A
import cv2

def build_transforms(args):
    data_transforms = {
        "train": A.Compose([
                A.ToFloat(max_value=255.0, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=20, p=0.4),
                A.Resize(args.image_size, args.image_size, always_apply=True),
                # ToTensorV2(always_apply=True),
            ]),

        "valid_test": A.Compose([
            A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
    return data_transforms




