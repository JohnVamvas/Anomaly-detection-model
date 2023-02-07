import tensorflow
import keras
import numpy
import cv2

from keras.models import load_model

import matplotlib.pyplot as plt

from LISTA import build_lista

from PSNR import psnr
from eta import EtaLayer

modelfile = "anomaly_detection.h5"

train = 0

latent_dim = 392
num_epochs = 10

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

def loss_graph(model):
    print(model.history.keys())
    # visualizing losses
    train_loss = model.history['loss']
    xc = range(num_epochs)

    plt.figure()
    plt.plot(xc, train_loss)
    plt.show()

def build_VAE_encoder_decoder():
    from VAE import build_VAE_model
    vae_build = build_VAE_model()
    vae_build.load_weights("VAE.h5")
    from VAE import encoder, decoder
    return encoder, decoder, vae_build

def make_z_dataset(vae_encoder, train_data): ##encoder
    z_training = vae_encoder.predict(train_data)[2]
    return z_training

def training(train_data):
    model = build_lista(latent_dim, (latent_dim * 2, latent_dim), (latent_dim * 2, latent_dim * 2), 3)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="mean_squared_error")
    seqModel = model.fit(train_data, train_data, epochs=num_epochs, batch_size=128, shuffle=True, verbose=2)
    loss_graph(seqModel)
    model.save(modelfile)

def testing(vae_encoder, vae_decoder):
    lista_vae_model = load_model(modelfile, custom_objects={'EtaLayer': EtaLayer})

    sum_average_l2_loss_lista_vae = 0
    mse = tensorflow.keras.losses.MeanSquaredError()

    sum_average_psnr_decoder = 0
    sum_average_l2_loss_decoder = 0

    test_normal_data = numpy.load('testing_normal_dataset.npy')
    test_anomalous_data = numpy.load('testing_full_corrupted_dataset.npy')

    test_array = numpy.concatenate((test_normal_data,test_anomalous_data)) #0...999 normal, 1000...1999 anomalous

    true_positives_1 = 0
    false_negatives_1 = 0
    false_positives_1 = 0
    true_negatives_1 = 0

    true_positives_2 = 0
    false_negatives_2 = 0
    false_positives_2 = 0
    true_negatives_2 = 0

    true_positives_3 = 0
    false_negatives_3 = 0
    false_positives_3 = 0
    true_negatives_3 = 0

    true_positives_4 = 0
    false_negatives_4 = 0
    false_positives_4 = 0
    true_negatives_4 = 0

    z_th = 4.2e-05
    psnr_th = 24.5

    for i in range(0, test_array.shape[0]):
        sum_l2_loss_lista_vae = 0
        sum_psnr_decoder = 0
        sum_l2_loss_decoder = 0
        for j in range(0, test_array[i].shape[0]):
            x_test_image = test_array[i][j]
            x_test_image = x_test_image.reshape(1, -1)

            z_test = vae_encoder.predict(x_test_image)[2]

            z_hat_test = lista_vae_model.predict(z_test)
            l2_loss_lista_vae = mse(z_test, z_hat_test).numpy()
            sum_l2_loss_lista_vae += l2_loss_lista_vae

            decode_image = vae_decoder.predict(z_test)
            test_psnr = psnr(x_test_image, decode_image)
            sum_psnr_decoder += test_psnr
            l2_loss_decoder = mse(x_test_image, decode_image).numpy()
            sum_l2_loss_decoder += l2_loss_decoder


        average_l2_loss_lista_vae = sum_l2_loss_lista_vae / (test_array[i].shape[0])
        sum_average_l2_loss_lista_vae += average_l2_loss_lista_vae

        average_psnr_decoder = sum_psnr_decoder / (test_array[i].shape[0])
        sum_average_psnr_decoder += average_psnr_decoder

        average_l2_loss_decoder = sum_l2_loss_decoder / (test_array[i].shape[0])
        sum_average_l2_loss_decoder += average_l2_loss_decoder

        criterion_1 = 0
        criterion_2 = 0
        if i<1000:
            if average_l2_loss_lista_vae>z_th: ##anomalous
                criterion_1 = 1
                false_positives_1 += 1
            else:
                true_negatives_1 += 1

            if average_psnr_decoder<psnr_th: ##amomalous
                criterion_2 = 1
                false_positives_2 += 1
            else:
                true_negatives_2 += 1

            if criterion_1 or criterion_2:##anomalous
                false_positives_3 += 1
            else:
                true_negatives_3 += 1

            if criterion_1 and criterion_2:##anomalous
                false_positives_4 += 1
            else:
                true_negatives_4 += 1
        else:
            if average_l2_loss_lista_vae>z_th: ##anomalous
                criterion_1 = 1
                true_positives_1 +=1
            else:
                false_negatives_1 += 1

            if average_psnr_decoder<psnr_th: ##anomalous
                criterion_2 = 1
                true_positives_2 +=1
            else:
                false_negatives_2 += 1

            if criterion_1 or criterion_2:##anomalous
                true_positives_3 += 1
            else:
                false_negatives_3 += 1

            if criterion_1 and criterion_2:##anomalous
                true_positives_4 += 1
            else:
                false_negatives_4 += 1

    print("VAE_LISTA_l2_loss")
    overall_l2_loss_lista_vae = sum_average_l2_loss_lista_vae/ test_array.shape[0]
    print(overall_l2_loss_lista_vae)

    print("")
    print("VAE")
    overall_psnr_decoder = sum_average_psnr_decoder / test_array.shape[0]
    print(overall_psnr_decoder)

    overall_l2_loss_decoder = sum_average_l2_loss_decoder / test_array.shape[0]
    print(overall_l2_loss_decoder)

    print("")

    accuracy_1 = (true_positives_1+true_negatives_1)/(true_positives_1+false_positives_1+false_negatives_1+true_negatives_1)
    print("accuracy_1: " + str(accuracy_1))

    precision_1 = true_positives_1/(true_positives_1+false_positives_1)
    print("precision_1 " + str(precision_1))

    recall_1 = true_positives_1/(true_positives_1+false_negatives_1)
    print("recall_1: " + str(recall_1))

    f1_score_1 = (2*(precision_1*recall_1))/(precision_1+recall_1)
    print("f1_score_1: " + str(f1_score_1))

    print("")

    accuracy_2 = (true_positives_2 + true_negatives_2) / (true_positives_2 + false_positives_2 + false_negatives_2 + true_negatives_2)
    print("accuracy_2: " + str(accuracy_2))

    precision_2 = true_positives_2 / (true_positives_2 + false_positives_2)
    print("precision_2 " + str(precision_2))

    recall_2 = true_positives_2 / (true_positives_2 + false_negatives_2)
    print("recall_2: " + str(recall_2))

    f1_score_2 = (2 * (precision_2 * recall_2)) / (precision_2 + recall_2)
    print("f1_score_2: " + str(f1_score_2))

    print("")

    accuracy_3 = (true_positives_3 + true_negatives_3) / (true_positives_3 + false_positives_3 + false_negatives_3 + true_negatives_3)
    print("accuracy_3: " + str(accuracy_3))

    precision_3 = true_positives_3 / (true_positives_3 + false_positives_3)
    print("precision_3 " + str(precision_3))

    recall_3 = true_positives_3 / (true_positives_3 + false_negatives_3)
    print("recall_3: " + str(recall_3))

    f1_score_3 = (2 * (precision_3 * recall_3)) / (precision_3 + recall_3)
    print("f1_score_3: " + str(f1_score_3))

    print("")

    accuracy_4 = (true_positives_4 + true_negatives_4) / (true_positives_4 + false_positives_4 + false_negatives_4 + true_negatives_4)
    print("accuracy_4: " + str(accuracy_4))

    precision_4 = true_positives_4 / (true_positives_4 + false_positives_4)
    print("precision_4 " + str(precision_4))

    recall_4 = true_positives_4 / (true_positives_4 + false_negatives_4)
    print("recall_4: " + str(recall_4))

    f1_score_4 = (2 * (precision_4 * recall_4)) / (precision_4 + recall_4)
    print("f1_score_4: " + str(f1_score_4))

if __name__ == "__main__":
    vae_encoder, vae_decoder, vae = build_VAE_encoder_decoder()
    if train:
        train_data = load_moving_MNIST()
        z_training_dataset = make_z_dataset(vae_encoder,train_data)
        training(z_training_dataset)
    else:
        testing(vae_encoder,vae_decoder)