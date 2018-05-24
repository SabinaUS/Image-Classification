import urllib.request
import os
import tarfile
import _pickle as cPickle
import numpy as np

def maybe_download(target_dir):
    cifar_data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar_data_file = os.path.join(target_dir, 'cifar-10-python.tar.gz')
    if not os.path.exists(cifar_data_file):
        print('Downloading cifar dataset...')
        urllib.request.urlretrieve(cifar_data_url, cifar_data_file)
    
        print('Extracting cifar dataset...')
        tar = tarfile.open(cifar_data_file, "r:gz")
        tar.extractall(target_dir)
        tar.close()    

class Cifar10(object):
    def __init__(self, train_batch_size, test_batch_size, data_format='NHWC'):
        self.data_format = data_format
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        download_dir = '../data'
        if not os.path.exists(download_dir):
            os.mkdir(download_dir)
        maybe_download(download_dir)
        
        self.cifar_dir = os.path.join(download_dir, 'cifar-10-batches-py')
        self.train_images, self.train_labels = self.load_train_data()
        self.test_images, self.test_labels = self.load_test_data()
        self.pp_mean = self.get_per_pixel_mean()
        self.shuffle_dataset()

        self.num_classes = 10
        if self.data_format == 'NHWC':
            self.input_shape = [32, 32, 3] 
        else:
            self.input_shape = [3, 32, 32]
        self.train_batch_count = self.train_images.shape[0] // self.train_batch_size
        self.test_batch_count = self.test_images.shape[0] // self.test_batch_size

    def load_train_data(self):
        train_files = ['data_batch_1',
                       'data_batch_2',
                       'data_batch_3',
                       'data_batch_4',
                       'data_batch_5']
        images, labels = [], []

        for data_file in train_files:
            full_path = os.path.join(self.cifar_dir, data_file)
            with open(full_path, 'rb') as f:
                raw = cPickle.load(f, encoding='latin1')

            count = raw['data'].shape[0]
            if self.data_format == 'NHWC':  
                batch = np.transpose(raw['data'].reshape((count, 3, 32, 32)), (0, 2, 3, 1))
            else:
                batch = raw['data'].reshape((count, 3, 32, 32))
            images += list(batch)
            labels += raw['labels']
        return np.array(images).astype(np.float32), np.array(labels)

    def load_test_data(self):
        test_file = 'test_batch'
        images, labels = [], []

        full_path = os.path.join(self.cifar_dir, test_file)
        with open(full_path, 'rb') as f:
            raw = cPickle.load(f, encoding='latin1')
    
        count = raw['data'].shape[0]
        if self.data_format == 'NHWC':  
            batch = np.transpose(raw['data'].reshape((count, 3, 32, 32)), (0, 2, 3, 1))
        else:
            batch = raw['data'].reshape((count, 3, 32, 32))
    
        images += (list(batch))
        labels += raw['labels']  
        return np.array(images).astype(np.float32), np.array(labels)

    def get_per_pixel_mean(self):
        images = np.concatenate((self.train_images, self.test_images), axis=0)
        return np.mean(images, axis=0)

    def shuffle_dataset(self):
        self.shuffle = np.random.permutation(self.train_images.shape[0])

    def normalize(self, batch_images):
        return (batch_images - self.pp_mean) / 128.0

    def next_train_batch(self, idx):
        batch_images = self.train_images[self.shuffle[idx * self.train_batch_size : (idx + 1) * self.train_batch_size]]
        batch_labels = self.train_labels[self.shuffle[idx * self.train_batch_size : (idx + 1) * self.train_batch_size]]
        return self.normalize(batch_images), batch_labels

    def next_test_batch(self, idx):
        batch_images = self.test_images[idx * self.test_batch_size : (idx + 1) * self.test_batch_size]
        batch_labels = self.test_labels[idx * self.test_batch_size : (idx + 1) * self.test_batch_size]
        return self.normalize(batch_images), batch_labels

    def next_aug_train_batch(self, idx):
        batch_images = self.train_images[self.shuffle[idx * self.train_batch_size : (idx + 1) * self.train_batch_size]]
        batch_labels = self.train_labels[self.shuffle[idx * self.train_batch_size : (idx + 1) * self.train_batch_size]]
        # Padding
        if self.data_format == 'NHWC':
            pad_width = ((0, 0), (4, 4), (4, 4), (0, 0))
        else:
            pad_width = ((0, 0), (0, 0), (4, 4), (4, 4))
        padded_images = np.pad(batch_images, pad_width, mode='constant', constant_values=0)
        # Random crop and flip
        aug_batch_images = np.zeros_like(batch_images)
        for i in range(len(batch_images)):
            x = np.random.randint(0, high=8)
            y = np.random.randint(0, high=8)
            if self.data_format == 'NHWC':
                cropped_img = padded_images[i][x : x + 32, y : y + 32, :]
            else:
                cropped_img = padded_images[i][:, x : x + 32, y : y + 32]
            is_flip = np.random.randint(0, high=3)
            #if is_flip % 2 == 0:
            #    flipped_img = np.flipud(cropped_img)
            #else:
            flipped_img = cropped_img
            np.copyto(aug_batch_images[i], flipped_img)
        
        return self.normalize(aug_batch_images), batch_labels


