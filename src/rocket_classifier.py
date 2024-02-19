import torch
from src.convolution import RocketFeatures
from src.classifier import LogisticRegression
from __init__ import volta_logger


def get_label_props(y):

    nb_obs = y.shape[0]
    return nb_obs / torch.sum(y, dim = 0)


class RocketClassifier:

    def __init__(self,
                 kernel_sizes,
                 dilations,
                 n_labels,
                 label_weights = None,
                 **kwargs):

        self.n_rfeats = 2 * len(kernel_sizes)
        self.n_labels = n_labels
        self.label_weights = label_weights

        self.rocket_feat_builder = RocketFeatures(kernel_sizes,
                                                  dilations,
                                                  **kwargs)

        self.classifier = LogisticRegression(self.n_rfeats, n_labels)
        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr = 1e-1)
        self.loss = torch.nn.CrossEntropyLoss(weight = self.label_weights,
                                              reduce = 'mean')

    def fit(self,
            y_train,
            X_train = None,
            rX_train = None,
            ks_print_freq = 100,
            loss_print_freq = 10,
            epochs = 50,
            logger = volta_logger,
            **kwargs):

        if (rX_train is None) and not (X_train is None):
            logger.info('Rocket features are not provided, yet, building them')
            rX_train = self.rocket_feat_builder(X_train,
                                                ks_print_freq)
        elif not (rX_train is None):
            logger.info('Rocket features already provided as training set')
        else:
            logger.error('Neither X_train nor rX_train given for the training aborting')
            return
        if self.label_weights is None:
            logger.info('No label weight provided for the cross entropy loss, estimating them using y_train')
            self.label_weights = torch.tensor(get_label_props(y_train),
                                              dtype = torch.float32,
                                              requires_grad = False)
            self.loss.weight = self.label_weights

        rX_train = torch.tensor(rX_train,
                                dtype = torch.float32,
                                requires_grad = False)

        y_train = torch.tensor(y_train,
                               dtype = torch.float32,
                               requires_grad = False)

        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            probs_train = self.classifier(rX_train)
            loss = self.loss(probs_train, y_train)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if epoch % loss_print_freq == 0:
                logger.info('Epoch: {}. Loss: {}'.format(epoch, loss.item()))

        return losses

    def predict(self,
                X_test = None,
                rX_test = None,
                ks_print_freq = 100,
                logger = volta_logger,
                **kwargs):

        if (rX_test is None) and not (X_test is None):
            logger.info('Rocket features are not provided, yet, building them')
            rX_test = self.rocket_feat_builder(X_test,
                                               ks_print_freq)
        elif not (rX_test is None):
            logger.info('Rocket features already provided')
        else:
            logger.error('Neither X_test nor rX_test given for prediction aborting')
            return

        rX_test = torch.tensor(rX_test,
                               dtype = torch.float32,
                               requires_grad = False)

        return self.classifier(rX_test)
