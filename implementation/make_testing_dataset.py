import keras
import numpy
import cv2

fpath = keras.utils.get_file(
    "moving_mnist.npy",
    "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
)
dataset = numpy.load(fpath)
print(dataset.shape)
dataset = numpy.swapaxes(dataset, 0, 1)
test_normal_dataset = dataset[8000:9000]
test_corrupted_dataset = dataset[9000:]
test_normal_dataset = test_normal_dataset / 255
test_corrupted_dataset = test_corrupted_dataset / 255

data_corrupted = numpy.empty((len(test_corrupted_dataset), dataset.shape[1], 28, 28), dtype=test_corrupted_dataset.dtype)
data_normal = numpy.empty((len(test_normal_dataset), dataset.shape[1], 28, 28), dtype=test_corrupted_dataset.dtype)

for i in range(0, len(test_corrupted_dataset)):
    for j in range(0, (test_corrupted_dataset[i].shape[0])):
        x_test_corrupted_image = cv2.resize(numpy.copy(test_corrupted_dataset[i][j]), (28, 28), interpolation=cv2.INTER_NEAREST)
        sample_1 = numpy.random.randint(low=3, high=25, size=(2,))
        position_i = sample_1[0]
        position_j = sample_1[1]
        x_test_corrupted_image[position_i][position_j] = 1
        for h in range(position_i - 3, position_i + 2):
            for f in range(position_j - 3, position_j + 2):
                x_test_corrupted_image[h][f] = 1
        data_corrupted[i][j] = x_test_corrupted_image

        data_normal[i][j] = cv2.resize(numpy.copy(test_normal_dataset[i][j]), (28, 28), interpolation=cv2.INTER_NEAREST)


numpy.save('testing_full_corrupted_dataset', data_corrupted)
numpy.save('testing_normal_dataset', data_normal)