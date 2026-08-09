import torch

from src.model import CatDogCNN


def test_model_output_shape():

    model = CatDogCNN()

    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    assert output.shape == (1, 2)