"""A module for providing tuning and training capabilities for Convolutional Neural Networks"""

from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dropout


class Conv2DBlock(Layer):
    def __init__(self, filters, kernel_size=(3, 3), activation='relu', padding='same',
                 max_pooling_size=(2, 2), dropout_rate=0.1):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.max_pooling_size = max_pooling_size
        self.dropout_rate = dropout_rate

        self.convolution = Conv2D(filters=filters, kernel_size=self.kernel_size,
                                  activation=self.activation, padding=self.padding)

        if self.max_pooling is not None:
            self.pooling = MaxPooling2D(self.max_pooling_size)

        if self.dropout_rate is not None:
            self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs):
        x = self.convolution(inputs)

        if self.max_pooling_size is not None:
            x = self.pooling(x)

        if self.dropout_rate is not None:
            x = self.dropout(x)

        return x
