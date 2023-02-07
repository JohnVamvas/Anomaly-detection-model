# Anomaly detection model
The purpose of this model was to train an Variational Autoencoder with a dataset (in this case was moving MNIST dataset). Then, we use the decoder and encoder of the previous variational autoencoder and the weights of the trained Variational autoencoder. Also, we feed the encoder with the moving MNIST dataset and the result will be the training dataset of LISTA. After the training of LISTA (that takes few epochs), we can see if a set of pictures (or videos etc) have anomalies.

More can be seen in file thesis.pdf
