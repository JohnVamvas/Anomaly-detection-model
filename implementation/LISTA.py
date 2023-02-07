import tensorflow
import keras
import numpy

import cv2
import matplotlib.pyplot as plt

from keras.models import load_model

from eta import EtaLayer

modelfile = "LISTA.h5"

train = 0

latent_dim = 392
num_epochs = 10

def loss_graph(model):
    print(model.history.keys())
    # visualizing loss
    train_loss = model.history['loss']
    xc = range(num_epochs)

    plt.figure()
    plt.plot(xc, train_loss)
    plt.show()

def build_VAE_encoder_decoder():
    from VAE import build_VAE_model
    vae_build = build_VAE_model()
    from VAE import encoder, decoder
    vae_build.load_weights("VAE.h5")
    return encoder, decoder, vae_build

def make_z_dataset(encoder, train_data):
    z_training = encoder.predict(train_data)[2]
    return z_training

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

    # Normalize the data to the 0-1 range.
    data = data / 255

    data = data.reshape(-1,64,64)
    resized_dataset = numpy.array([cv2.resize(numpy.copy(d), (28, 28),interpolation=cv2.INTER_NEAREST) for d in data])
    train_dataset = resized_dataset

    x_train = numpy.reshape(train_dataset, newshape=(train_dataset.shape[0], numpy.prod(train_dataset.shape[1:])))

    return x_train


def build_lista(ylen, wdim, sdim, numlayers):
    print("\tConstructing LISTA autoencoder ...")

    y_ = keras.Input(shape=(ylen,))
    print("\t>>> input y")
    print("\t", y_)

    Wy_ = keras.layers.Dense(wdim[0])(y_)
    print("\t>>> layer Wy_")
    print("\t", y_)

    uhat_ = EtaLayer()(Wy_)
    print("\t>>> uhat0_")
    print("\t", uhat_)

    for t in range(1, numlayers):
        print("\tLayer " + str(t))

        Suhat_ = keras.layers.Dense(sdim[0])(uhat_)

        added_ = keras.layers.Add()([Suhat_, Wy_])

        uhat_ = EtaLayer()(added_)
        print("\t>>> uhat" + str(t) + "_")
        print("\t", uhat_)

    rec_ = keras.layers.Dense(ylen)(uhat_)
    print("\t>>> reconstruction layer")
    print("\t", rec_)

    model = keras.models.Model(inputs=y_, outputs=rec_)
    model.summary()

    return model

def training(model,train_data):
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="mean_squared_error")
    seqModel = model.fit(train_data, train_data, epochs=num_epochs, batch_size=128, shuffle=True, verbose=2)
    loss_graph(seqModel)
    model.save(modelfile)

def testing(vae_encoder):
    model = load_model(modelfile, custom_objects={'EtaLayer': EtaLayer})

    mse = tensorflow.keras.losses.MeanSquaredError()

    test_normal_data = numpy.load('testing_normal_dataset.npy')

    overall_average_mse = 0

    for i in range(0, test_normal_data.shape[0]):
        sum_mse = 0
        for j in range(0, (test_normal_data[i].shape[0])):
            x_test_image = test_normal_data[i][j]
            x_test_image = x_test_image.reshape(1, -1)

            z_test = vae_encoder.predict(x_test_image)[2]
            z_hat_test = model.predict(z_test)

            l2_loss = mse(z_test, z_hat_test).numpy()
            sum_mse += l2_loss

        average_mse = sum_mse/test_normal_data[i].shape[0]
        overall_average_mse += average_mse

    overall_mse = overall_average_mse / test_normal_data.shape[0]
    print("overall_mse: " + str(overall_mse))

if __name__ == "__main__":
    vae_encoder, vae_decoder, vae = build_VAE_encoder_decoder()
    if train:
        train_data = load_moving_MNIST()
        z_training_dataset = make_z_dataset(vae_encoder,train_data)
        model = build_lista(latent_dim, (latent_dim*2, latent_dim), (latent_dim*2, latent_dim*2), 3)
        training(model, z_training_dataset)
    else:
        testing(vae_encoder)