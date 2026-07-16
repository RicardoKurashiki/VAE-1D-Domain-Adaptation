from torchvision import transforms
from utils.add_random_noise import AddRandomNoise


# Configuração de normalização/entrada (ResNet-18, ImageNet)
MODEL_CONFIGS = {
    "default": {
        "input_size": 224,
        "resize_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def get_model_config(model_name):
    """Retorna a configuração específica do modelo ou a configuração padrão."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])


def build_augmentation_transforms(config, is_training=True, model_name=None):
    aug_config = config.get('data_augmentation', {})
    use_augmentation = config.get('use_data_augmentation', False) and is_training

    if model_name is None:
        model_name = config.get('model', 'default')

    model_config = get_model_config(model_name)
    input_size = model_config["input_size"]
    resize_size = model_config["resize_size"]

    transform_list = []

    zoom_range = aug_config.get('zoom_range')
    if zoom_range is None:
        zoom_range = [1.0, 1.0]
    elif isinstance(zoom_range, (int, float)):
        zoom_range = [zoom_range, 1.0]
    if use_augmentation and zoom_range != [1.0, 1.0]:
        transform_list.append(
            transforms.RandomResizedCrop(input_size, scale=(zoom_range[0], zoom_range[1]))
        )
    else:
        transform_list.append(transforms.Resize(resize_size))
        transform_list.append(transforms.CenterCrop(input_size))

    if use_augmentation:
        rotation_range = aug_config.get('rotation_range')
        if rotation_range is None:
            rotation_range = 0
        if rotation_range > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=(-rotation_range, rotation_range))
            )

        if aug_config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if aug_config.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))

        shear_range = aug_config.get('shear_range')
        if shear_range is None:
            shear_range = 0
        width_shift = aug_config.get('width_shift_range')
        if width_shift is None:
            width_shift = 0
        height_shift = aug_config.get('height_shift_range')
        if height_shift is None:
            height_shift = 0

        if shear_range > 0 or width_shift > 0 or height_shift > 0:
            translate = (width_shift, height_shift) if (width_shift > 0 or height_shift > 0) else None
            shear = (-shear_range, shear_range) if shear_range > 0 else None
            transform_list.append(
                transforms.RandomAffine(degrees=0, translate=translate, shear=shear)
            )

        brightness_range = aug_config.get('brightness_range')
        if brightness_range is None:
            brightness_range = [1.0, 1.0]
        contrast_range = aug_config.get('contrast_range')
        if contrast_range is None:
            contrast_range = [1.0, 1.0]

        brightness_factor = None
        if brightness_range != [1.0, 1.0]:
            brightness_factor = brightness_range

        contrast_factor = None
        if contrast_range != [1.0, 1.0]:
            contrast_factor = contrast_range

        if brightness_factor or contrast_factor:
            transform_list.append(
                transforms.ColorJitter(brightness=brightness_factor, contrast=contrast_factor)
            )

    transform_list.append(transforms.ToTensor())

    if use_augmentation:
        noise_std_range = aug_config.get('random_noise_std_range')
        if noise_std_range is not None:
            transform_list.append(AddRandomNoise(std_range=tuple(noise_std_range)))

    transform_list.append(
        transforms.Normalize(mean=model_config["mean"], std=model_config["std"])
    )

    return transforms.Compose(transform_list)


def get_validation_transforms(model_name="default"):
    model_config = get_model_config(model_name)
    input_size = model_config["input_size"]
    resize_size = model_config["resize_size"]

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_config["mean"], std=model_config["std"])
    ])
