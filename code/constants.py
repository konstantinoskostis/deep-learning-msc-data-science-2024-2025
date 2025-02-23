"""A set of constants as sane defaults"""

TUNERS_DIR = 'tuners'
MODELS_DIR = 'models'

FASHION_MNIST_TRAIN_IMAGES_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
FASHION_MNIST_TRAIN_LABELS_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
FASHION_MNIST_TEST_IMAGES_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
FASHION_MNIST_TEST_LABELS_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
FASHION_MNIST_DATA_PATH = 'data/fashion_mnist'
FASHION_MNIST_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FASHION_MNIST_CLASS_LABELS = [
    'T-shirt/top', 'Trouser', 'Pullover',
    'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot'
]
FASHION_MNIST_MLP_TUNER_PROJECT_NAME = 'fashion_mninst_mlp'
FASHION_MNIST_MLP_WEIGHTS = 'fashion_mnist_mlp.weights.h5'
FASHION_MNIST_CNN_TUNER_PROJECT_NAME = 'fashion_mninst_cnn'
FASHION_MNIST_CNN_WEIGHTS = 'fashion_mnist_cnn.weights.h5'

CIFAR_10_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CIFAR_10_CLASS_LABELS = [
    'airplane', 'automobile', 'bird',
    'cat', 'deer', 'dog', 'frog',
    'horse', 'ship', 'truck'
]
CIFAR_10_MLP_TUNER_PROJECT_NAME = 'cifar_10_mlp'
CIFAR_10_MLP_WEIGHTS = 'cifar_10_mlp.weights.h5'
CIFAR_10_CNN_TUNER_PROJECT_NAME = 'cifar_10_cnn'
CIFAR_10_CNN_WEIGHTS = 'cifar_10_cnn.weights.h5'

SEED = 2025
