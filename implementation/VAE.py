import tensorflow
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets

import matplotlib.pyplot as plt

import numpy
import cv2

import keras
from keras.regularizers import l2

from PSNR import psnr

global z_log_var
global z_mean
global encoder
global decoder

modelfile = "VAE.h5"

train = 0

original_dim = 784 #(28x28)
latent_dim = 392
num_epochs = 100

def load_moving_MNIST():
    # Download and load the dataset.
    fpath = keras.utils.get_file(
        "moving_mnist.npy",
        "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
    )
    dataset = numpy.load(fpath)
    dataset = numpy.swapaxes(dataset, 0, 1)
    sample = numpy.random.randint(low=0, high=8000, size=(5000,))
    data = numpy.empty((len(sample),dataset.shape[1],dataset.shape[2],dataset.shape[2]),dtype=dataset.dtype)

    for i in range(len(sample)):
        data[i] = dataset[sample[i]]

    data = data.reshape(-1,64,64)

    # Normalize the data to the 0-1 range.
    data = data / 255

    resized_dataset = numpy.array([cv2.resize(numpy.copy(d), (28, 28),interpolation=cv2.INTER_NEAREST) for d in data])
    train_dataset = resized_dataset

    x_train = numpy.reshape(train_dataset, newshape=(train_dataset.shape[0], numpy.prod(train_dataset.shape[1:])))

    return x_train

def loss_graph(model):
    print(model.history.keys())
    # visualizing losses and accuracy
    train_loss = model.history['loss']
    xc = range(num_epochs)

    plt.figure()
    plt.plot(xc, train_loss)
    plt.show()

def sampling(z_sampling):
    z_mean = z_sampling[0]
    z_log_sigma = z_sampling[1]
    epsilon = tensorflow.keras.backend.random_normal(shape=(tensorflow.keras.backend.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + tensorflow.keras.backend.exp(z_log_sigma) * epsilon

def build_VAE_model():
    global z_log_var
    global z_mean
    global encoder
    global decoder

    # encoder
    input = tensorflow.keras.layers.Input(shape=(original_dim,), name="encoder_input")

    encoder_dense_layer1 = tensorflow.keras.layers.Dense(units=400, activation='relu', kernel_regularizer=l2(0.001),bias_regularizer=l2(0.001), name="encoder_dense_1")(input)

    z_mean = tensorflow.keras.layers.Dense(latent_dim, name="z_mean")(encoder_dense_layer1)
    z_log_var = tensorflow.keras.layers.Dense(latent_dim, name="z_log_var")(encoder_dense_layer1)

    z = tensorflow.keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = tensorflow.keras.models.Model(input, [z_mean, z_log_var, z], name="encoder_model")
    encoder.summary()

    # Decoder
    decoder_input = tensorflow.keras.layers.Input(shape=(latent_dim,), name="decoder_input")

    decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=400, activation='relu', kernel_regularizer=l2(0.001),bias_regularizer=l2(0.001), name="decoder_dense_1")(decoder_input)

    decoder_output = tensorflow.keras.layers.Dense(units=original_dim, activation='sigmoid', name="decoder_dense_2")(decoder_dense_layer1)

    decoder = tensorflow.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
    decoder.summary()

    # Variational_Autoencoder
    z_space = encoder(input)[2]
    vae_decoder_output = decoder(z_space)

    vae = tensorflow.keras.models.Model(inputs=input, outputs=vae_decoder_output, name="AE")
    vae.summary()

    return vae

@tensorflow.function
def nll(y_true, y_pred):
    # """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean

    reconstruction_loss = tensorflow.keras.losses.binary_crossentropy(y_true, y_pred)
    reconstruction_loss *= original_dim
    reconstruction_loss *= 1 / 0.7

    kl_loss = 1 + z_log_var - tensorflow.keras.backend.square(z_mean) - tensorflow.keras.backend.exp(z_log_var)
    kl_loss = tensorflow.keras.backend.sum(kl_loss, axis=-1)  # axis = -1
    kl_loss *= -0.5

    return tensorflow.keras.backend.mean(kl_loss + reconstruction_loss)  #50% kl_loss + 50% reconstruction_loss


def training(model,train_data):
    # VAE Compilation
    # lr---->learning rate
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001), loss=nll, experimental_run_tf_function=False)

    # Training VAE
    seqModel = model.fit(train_data, train_data, epochs=num_epochs, batch_size=128, shuffle=True,verbose=2)
    loss_graph(seqModel)
    model.save(modelfile)

def testing():
    model = tensorflow.keras.models.load_model(modelfile,custom_objects={"sampling": sampling},compile=False)

    test_normal_data = numpy.load('testing_normal_dataset.npy')

    sum_average_psnr = 0

    for i in range(0, test_normal_data.shape[0]):
        sum_psnr = 0
        for j in range(0, test_normal_data[i].shape[0]):
            x_test_image = test_normal_data[i][j]
            x_test_image = x_test_image.reshape(1, -1)

            decoded_images = model.predict(x_test_image)  # encoder

            test_psnr = psnr(x_test_image, decoded_images)
            sum_psnr += test_psnr

        average_psnr = sum_psnr / (test_normal_data[i].shape[0])
        sum_average_psnr += average_psnr

    overall_psnr = sum_average_psnr/test_normal_data.shape[0]
    print("overall_psnr: " + str(overall_psnr))

if __name__ == "__main__":
    if train:
        train_data= load_moving_MNIST()
        model = build_VAE_model()
        training(model,train_data)
    else:
        testing()