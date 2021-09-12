import feacher


# Path/Directory containg images
path = 'tests/images/'

def test_basic():
    # Extract features using Feacher
    image_features = feacher.extract(path)

    assert image_features[0].shape == (1, 512, 1, 1)

def test_vgg16():
    # Extract features using Feacher
    image_features = feacher.extract(path,
                                 pretrained_model='vgg16',
                                 resize_dim=256,
                                 layer='features[30]')

    assert image_features[0].shape == (1, 512, 7, 7)
