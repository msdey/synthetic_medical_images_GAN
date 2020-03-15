from train import *

# Generator #
def gen_model(noise_shape = noise_shape):
    noise_shape = noise_shape
    layer_filters = [ 1024,512, 256, 128,64,32,16, 8, 4,1]
    kernel_size = 5
    inputs = Input(shape = noise_shape)

    x = inputs
    for filters in layer_filters:
      if filters > layer_filters[-2]:
        strides  = 2
      else:
        strides = 1

      x = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(x)
      x = BatchNormalization(momentum = 0.5)(x)
      x = LeakyReLU(0.2)(x)

    outs = Activation('sigmoid')(x)
        
    optimizer = Adam(lr=0.0001)
    model = Model(input = inputs, output = outs)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


# Discriminator #
def disc_model(image_shape=(256,256,1)):
    image_shape = image_shape
    
    dropout_prob = 0.4
    inputs = Input(shape = image_shape)
    x = inputs
    kernel_size = 5
    layer_filters = [16, 32, 64, 128, 256, 512]
    for filters in layer_filters:
      strides = 2
      x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(x)
      x = BatchNormalization(momentum = 0.5)(x)
      x = LeakyReLU(0.2)(x)

    
    x = Flatten()(x)
    x = Dense(1)(x)
    outs = Activation('sigmoid')(x)


    optimizer = optimizers.SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
    model = Model(input = inputs, output = outs)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

discriminator = disc_model()
generator = gen_model()

discriminator.trainable = False

# Generative Adversarial Network #

inputs = Input(shape=noise_shape)
gan = Model(input = inputs, output = discriminator(generator(inputs)))
gan.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.0001), metrics=['accuracy'])
gan.summary()

