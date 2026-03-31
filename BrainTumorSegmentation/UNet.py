import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
)



class UNet:
    def __init__(self, input_size=(256, 256, 3), base_filters=32):
        self.input_size = input_size
        self.base_filters = base_filters
        self.model = self._build_model()

    def conv_block(self, x, filters):
        x = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        return x

    def _build_model(self):
        inputs = Input(shape=self.input_size)

        c1 = self.conv_block(inputs, self.base_filters)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = self.conv_block(p1, self.base_filters * 2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = self.conv_block(p2, self.base_filters * 4)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = self.conv_block(p3, self.base_filters * 8)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = self.conv_block(p4, self.base_filters * 16)

        u6 = Conv2DTranspose(self.base_filters * 8, (2, 2), strides=(2, 2), padding="same")(c5)
        u6 = concatenate([u6, c4], axis=-1)
        c6 = self.conv_block(u6, self.base_filters * 8)

        u7 = Conv2DTranspose(self.base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c6)
        u7 = concatenate([u7, c3], axis=-1)
        c7 = self.conv_block(u7, self.base_filters * 4)

        u8 = Conv2DTranspose(self.base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c7)
        u8 = concatenate([u8, c2], axis=-1)
        c8 = self.conv_block(u8, self.base_filters * 2)

        u9 = Conv2DTranspose(self.base_filters, (2, 2), strides=(2, 2), padding="same")(c8)
        u9 = concatenate([u9, c1], axis=-1)
        c9 = self.conv_block(u9, self.base_filters)

        outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

        return Model(inputs=inputs, outputs=outputs, name="U_Net")

def save_model_summary_with_params(model, filename="model_summary.png"):
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    summary_text = "\n".join(summary_lines)

    total_params = model.count_params()

    fig = plt.figure(figsize=(14, 10))
    plt.axis("off")

    full_text = (
        f"Model Summary\n"
        f"Total parameters: {total_params:,}\n\n"
        f"{summary_text}"
    )

    plt.text(
        0.01,
        0.99,
        full_text,
        fontsize=10,
        va="top",
        family="monospace",
    )

    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close(fig)
    
def main():
    unet = UNet()
    model = unet.model
    save_model_summary_with_params(model, "graphics/unet_summary.png")

if __name__ == '__main__':
    main()