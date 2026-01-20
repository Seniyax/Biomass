class AgumentationFactory:
    def __init__(self,img_size):
        self.img_size = img_size

    def train_transform(self) -> A.Compose:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.75

            ),


            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])  

    def valid_transform(self) -> A.Compose:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]) 