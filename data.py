import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import torch
from torchvision.datasets import MNIST, SVHN, CIFAR10
from torchvision import transforms
import torchvision.utils as vutils

class DataLoader(object):

    def __init__(self, config, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()

        if config.dataset == 'mnist':
            self.images = self.images.view(self.images.size(0), -1)

        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def get_zca_cuda(self, reg=1e-6):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        mean = images.mean(0)
        images -= mean.expand_as(images)
        sigma = torch.mm(images.transpose(0, 1), images) / images.size(0)
        U, S, V = torch.svd(sigma)
        components = torch.mm(torch.mm(U, torch.diag(1.0 / torch.sqrt(S) + reg)), U.transpose(0, 1))
        return components, mean

    def apply_zca_cuda(self, components):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        self.images = torch.mm(images, components.transpose(0, 1)).cpu()

    def generator(self, inf=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ind = torch.LongTensor(list(indices[start: end]))
                ret_images, ret_labels = self.images[ind], self.labels[ind]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len

def get_twomoons_loaders(config):
    # Generate data
    num_data = config.size_labeled_data + config.size_unlabeled_data + config.size_dev
    data,labels = datasets.make_moons((round(num_data/2),round(num_data/2)), noise = config.data_noise, random_state = 42)
    dataset = []
    for i in range(len(data)):
        dataset.append([torch.Tensor(data[i]), labels[i]])
    dataset = np.array(dataset,dtype=object)
    
    # Train val split
    msk = np.random.rand(len(dataset)) < 1.*config.size_dev/num_data
    training_set = dataset[list(msk)]
    dev_set = dataset[list(~msk)]

    # Labeled and unlabeled split
    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    
    for i in range(config.num_label):
        mask[np.where(labels == i)[0][: round(config.size_labeled_data / config.num_label)]] = True
    
    labeled_indices, unlabeled_indices = indices[mask], indices[~mask]
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    if(config.visualize):
        x = np.array([dataset[:,0][i][0] for i in range(len(dataset))])
        y = np.array([dataset[:,0][i][1] for i in range(len(dataset))])
        fig, ax = plt.subplots()
        for l in range(config.num_label):
            xl = x[dataset[:,1]==l]
            yl = y[dataset[:,1]==l]
            plt.plot(xl,yl,'*',label=l)
            plt.plot(x[labeled_indices],y[labeled_indices],'k*')

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(config.num_label):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set, dataset

