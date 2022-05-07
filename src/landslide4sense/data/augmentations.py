import albumentations as A

h, w = 128, 128


transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.RandomSizedCrop(min_max_height=[96, 96], height= 128, width=128, p=0.5),
        A.ShiftScaleRotate(p=0.5)
    ]
)
