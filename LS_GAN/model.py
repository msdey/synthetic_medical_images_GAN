
from preprocessing import *
# Generator #

def gen_model(noise_shape = noise_shape):
    noise_shape = noise_shape
    layer_filters = [  512, 256, 128,64,32,16, 1]
    kernel_size = 3
    inputs = Input(shape = noise_shape)
    image_resize = 8
    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
      if filters > layer_filters[-2]:
        strides  = 2
      else:
        strides = 1

      x = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(x)
      x = BatchNormalization(momentum = 0.5)(x)
      x = LeakyReLU(0.2)(x)


    outs = Activation('tanh')(x)

    model = Model(input = inputs, output = outs)

    return model

gen = gen_model()
for l in gen.layers:
   print (l.output_shape)


# Discriminator #
def disc_model(image_shape=image_shape):
    image_shape = image_shape
    
    inputs = Input(shape = image_shape)
    x = inputs
    kernel_size = 3
    layer_filters = [16, 32, 64, 128, 256]
    for filters in layer_filters:
      strides = 2
      x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(x)
      x = BatchNormalization(momentum = 0.5)(x)
      x = LeakyReLU(0.2)(x)

    
    x = Flatten()(x)
    outs = Dense(1)(x)

    model = Model(input = inputs, output = outs)
    model.compile(loss = 'mse', optimizer = Adam(lr = 0.00001, beta_1 = 0.5), metrics=['accuracy'])

    return model

discriminator = disc_model()
generator = gen_model()

discriminator.trainable = False

# LSGAN #
inputs = Input(shape=noise_shape)
gan = Model(input = inputs, output = discriminator(generator(inputs)))
gan.compile(loss = 'mse', optimizer = Adam(lr = 0.00001, beta_1 = 0.5), metrics=['accuracy'])
gan.summary()


