"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        self.embedding_sharing = embedding_sharing
        self.sparse = sparse

        self.U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
        self.Q = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim, sparse=sparse)
        self.A = ZeroEmbedding(num_embeddings=num_users, embedding_dim=1)
        self.B = ZeroEmbedding(num_embeddings=num_items, embedding_dim=1)

        self.U_reg, self.Q_reg = None, None

        if embedding_sharing:
            self.U_reg = self.U
            self.Q_reg = self.Q
        else:
            self.U_reg = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
            self.Q_reg = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim, sparse=sparse)

        self.layers = nn.Sequential()
        for i, _ in enumerate(layer_sizes):
            if i == len(layer_sizes) - 1:
                self.layers.add_module("linear_" + str(i), nn.Linear(layer_sizes[i], 1))
            else:
                self.layers.add_module("linear_" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                self.layers.add_module("relu_" + str(i), nn.ReLU())

        # ********************************************************
        # ********************************************************
        # ********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        U = self.U(user_ids)
        Q = self.Q(item_ids)
        A = torch.squeeze(self.A(user_ids), dim=-1)
        B = torch.squeeze(self.B(user_ids), dim=-1)
        p_ij = torch.sum(U * Q, dim=-1)
        if self.sparse:
            p_ij = torch.diag(p_ij)
        predictions = p_ij + A + B

        U = self.U_reg(user_ids)
        Q = self.Q_reg(item_ids)
        x = torch.cat((U, Q, U * Q), dim=-1)
        score = torch.squeeze(self.layers(x), dim=-1)

        # ********************************************************
        # ********************************************************
        # ********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")

        return predictions, score
