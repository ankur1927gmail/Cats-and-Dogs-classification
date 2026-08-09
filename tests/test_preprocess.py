from PIL import Image

from src.preprocess import train_transform


def test_train_transform_returns_tensor():

    image = Image.new("RGB", (300, 300))

    transformed_image = train_transform(image)

    assert transformed_image.shape[0] == 3
    assert transformed_image.shape[1] == 224
    assert transformed_image.shape[2] == 224