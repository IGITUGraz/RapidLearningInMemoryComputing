import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import random

def make_shuffle_index(rng, num_classes, num_shots):
    a = np.arange(num_classes).repeat(num_shots).reshape((num_classes, num_shots)).T
    a = rng.permuted(a, axis=1)
    return a.flatten() + np.repeat(np.arange(num_shots) * num_classes, num_classes)


class OmniglotV2:
    def __init__(self, num_test, num_train, inner_classes, seed, height_width=28):
        self.num_train = num_train
        self.num_test = num_test
        self.num_classes = inner_classes
        self.num_test_all = self.num_test * self.num_classes
        self.num_train_all = self.num_train * self.num_classes
        self.n_out = inner_classes
        self.num_classes_all = 1623
        self.height_width = 28
        self.image_shape = [self.height_width, self.height_width, 1]

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        train_ds = tfds.load('omniglot', split='train', as_supervised=True, shuffle_files=False)
        test_ds = tfds.load('omniglot', split='test', as_supervised=True, shuffle_files=False)
        self.train_data = {}
        self.test_data = {}

        self.train_labels = []
        self.test_labels = []

        def extract(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize(image, [28, 28])
            return image, label

        for ds, data, labels in [[train_ds, self.train_data, self.train_labels], [test_ds, self.test_data, self.test_labels]]:
            for image, label in ds.map(extract):
                image = image.numpy()
                label = str(label.numpy())
                if label not in data:
                    data[label] = []
                data[label].append(image)
            labels.extend(list(data.keys()))

    @staticmethod
    def get_loss(output, targets):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(targets, output))
        if tf.math.is_nan(loss):
            import pudb
            pu.db

        return loss

    @staticmethod
    def get_evaluation(output, targets):
        acc = tf.reduce_mean(tf.where(tf.argmax(output, axis=-1) == targets, 1.0, 0.0))

        return acc
    def make_np_batch(self, batch_size, training=True, shuffle=True, collapse=True):
        if training:
            data = self.train_data
            labels = self.train_labels
        else:
            data = self.test_data
            labels = self.test_labels

        train_images = np.zeros((batch_size, self.num_train_all, 28, 28, 1), dtype=np.float32)
        train_labels = np.zeros((batch_size, self.num_train_all), dtype=np.int64)
        test_images = np.zeros((batch_size, self.num_test_all, 28, 28, 1), dtype=np.float32)
        test_labels = np.zeros((batch_size, self.num_test_all), dtype=np.int64)

        for b in range(batch_size):
            label_subset = random.choices(labels, k=self.num_classes)

            for i in range(len(label_subset)):
                train_slice = slice(i * self.num_train, (i + 1) * self.num_train)
                test_slice = slice(i * self.num_test, (i + 1) * self.num_test)
                train_labels[b, train_slice] = i
                test_labels[b, test_slice] = i

                images_to_split = random.sample(data[label_subset[i]], k=self.num_train + self.num_test)
                if training:
                    images_to_split = np.rot90(images_to_split, k=random.choice([0, 1, 2, 3]), axes=[1, 2])
                train_images[b, train_slice] = images_to_split[:self.num_train]
                test_images[b, test_slice] = images_to_split[self.num_train]

            # transform labels (and data) [0, 0, 0, 1, 1, 1, ...] into [0, 1, 2, 0, 1, 2, ...]
            train_images[b] = np.transpose(train_images[b].reshape((self.num_classes, self.num_train, 28, 28, 1)), [1, 0, 2, 3, 4]).reshape((self.num_train_all, 28, 28, 1))
            train_labels[b] = np.transpose(train_labels[b].reshape((self.num_classes, self.num_train)), [1, 0]).reshape(self.num_train_all)
            test_images[b] = np.transpose(test_images[b].reshape((self.num_classes, self.num_test, 28, 28, 1)), [1, 0, 2, 3, 4]).reshape((self.num_test_all, 28, 28, 1))
            test_labels[b] = np.transpose(test_labels[b].reshape((self.num_classes, self.num_test)), [1, 0]).reshape(self.num_test_all)

            # shuffle data
            if shuffle:
                shuffle_index = make_shuffle_index(self.rng, self.num_classes, self.num_train)
                train_images[b] = train_images[b][shuffle_index]
                train_labels[b] = train_labels[b][shuffle_index]

                shuffle_index = make_shuffle_index(self.rng, self.num_classes, self.num_test)
                test_images[b] = test_images[b][shuffle_index]
                test_labels[b] = test_labels[b][shuffle_index]
        if collapse:
            images = np.concatenate([train_images, test_images], axis=1)
            labels = np.concatenate([train_labels, test_labels], axis=1)
            return images, labels

        return train_images, train_labels, test_images, test_labels
    
    def get_dataset(self, batch_size, train=True, shuffle=True):
        images, labels = self.make_np_batch(batch_size, training=train, shuffle=shuffle, collapse=True)
        images = tf.constant(images)
        labels = tf.constant(labels, dtype=tf.int64)
        return images, labels

if __name__ == '__main__':
    ds = OmniglotV2(5, 5, 5, 42)
    images, labels = ds.get_dataset(32)
    print(images.shape, labels.shape)
