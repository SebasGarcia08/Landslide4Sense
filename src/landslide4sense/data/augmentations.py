import albumentations as A

h, w = 128, 128


transforms = A.Compose(
    [
        A.OneOf(
            [
                A.RandomSizedCrop(min_max_height=(50, 101), height=h, width=w, p=0.5),
                A.PadIfNeeded(min_height=h, min_width=w, p=0.5),
            ],
            p=1,
        ),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.OneOf(
            [
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                ),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
            ],
            p=0.8,
        ),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8),
    ]
)
