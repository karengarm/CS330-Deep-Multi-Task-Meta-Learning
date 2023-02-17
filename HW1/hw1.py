import argparse

import os
import random
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from load_data import DataGenerator
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter
import torchvision


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        batch_size= input_images.shape[0]
        support_set_image = input_images[:, :-1, :, :]  # [B, K, N, 784] flattened images
        support_set_labels = input_labels[:, :-1, :, :]  # [B, K, N, N] ground truth labels
        query_set_images = input_images[:, -1:, :, :]   # [B, 1, N, 784] flattened images
        # Pass zeros, not the ground truth labels for the final N examples
        query_set_labels = torch.zeros_like(input_labels[:, -1:, :, :])  # [B, 1, N, N] labels

        # Each set of labels and images are concatenated together
        support_input = torch.cat((support_set_image, support_set_labels), dim=-1)  # [B, K+1, N, 784 + N]
        query_input = torch.cat((query_set_images, query_set_labels), dim=-1)  # [B, 1, N, 784 + N]

        input = torch.cat((support_input, query_input), dim=1)  # [B, K + 1, N, 784 + N]
        input = torch.reshape(input,
                              (batch_size, -1, 784 + self.num_classes))  # [B, (K+1) * N, 784 + N]

        # the Nx K support set examples are sequentially passed through the network
        input, _ = self.layer1(input)  # [B, K * N, size of the model]
        input, _ = self.layer2(input)  # [B, K * N, N]
        input = torch.reshape(input, shape=(batch_size, -1,  self.num_classes, self.num_classes))
        return input
        #############################

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################
        #### YOUR CODE GOES HERE ####
        # 2. Fill in the function called loss function in the MANN class which takes as
        preds = preds[:, -1, :, :]  # [B;K+1;N;N]
        preds = torch.reshape(preds, shape=(-1, self.num_classes))

        # input the [B;K+1;N;N] labels
        labels = labels[:, -1, :, :]
        labels = torch.argmax(labels, dim=-1)  # [B, N]
        labels = torch.reshape(labels, shape=(-1,))
        # loss is computed between the query set predictions and the ground truth labels
        loss = F.cross_entropy(preds, labels)
        return loss
        #############################


def train_step(images, labels, model, optim, eval=False):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()


def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(
        f"runs/{config.num_classes}_{config.num_shot}_{config.random_seed}_{config.hidden_dim}"
    )

    # Download Omniglot Dataset
    if not os.path.isdir("./omniglot_resized"):
        gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized.zip",
            unzip=True,
        )
    assert os.path.isdir("./omniglot_resized")

    # Create Data Generator
    train_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="train",
        device=device,
        cache=config.image_caching,
    )
    train_loader = iter(
        torch.utils.data.DataLoader(
            train_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    test_iterable = DataGenerator(
        config.num_classes,
        config.num_shot + 1,
        batch_type="test",
        device=device,
        cache=config.image_caching,
    )
    test_loader = iter(
        torch.utils.data.DataLoader(
            test_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    # Create model
    model = MANN(config.num_classes, config.num_shot + 1, config.hidden_dim)
    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    import time

    times = []
    for step in range(config.train_steps):
        ## Sample Batch
        t0 = time.time()
        i, l = next(train_loader)
        i, l = i.to(device), l.to(device)
        t1 = time.time()

        ## Train
        _, ls = train_step(i, l, model, optim)
        t2 = time.time()
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])

        ## Evaluate
        if (step + 1) % config.eval_freq == 0:
            print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, l = next(test_loader)
            i, l = i.to(device), l.to(device)
            pred, tls = train_step(i, l, model, optim, eval=True)
            print("Train Loss:", ls.cpu().numpy(), "Test Loss:", tls.cpu().numpy())
            writer.add_scalar("Loss/test", tls, step)
            pred = torch.reshape(
                pred, [-1, config.num_shot + 1, config.num_classes, config.num_classes]
            )
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            l = torch.argmax(l[:, -1, :, :], axis=2)
            acc = pred.eq(l).sum().item() / (config.meta_batch_size * config.num_classes)
            print("Test Accuracy", acc)
            writer.add_scalar("Accuracy/test", acc, step)

            times = np.array(times)
            print(f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}")
            times = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_shot", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_steps", type=int, default=25000)
    parser.add_argument("--image_caching", type=bool, default=True)
    main(parser.parse_args())
