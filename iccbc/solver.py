import datetime
import os
import time
from torch.nn.functional import binary_cross_entropy

from iccbc.utils import time_left

import torch

from tensorboardX import SummaryWriter


class Solver(object):

    def __init__(self):
        self.history = {}

        self.optim = []
        self.criterion = []
        self.training_time_s = 0
        self.epoch = 0

    def train(
        self,
        model,
        train_config,
        tensorboard_path,
        optim=None,
        num_epochs=10,
        max_train_time_s=None,
        train_loader=None,
        val_loader=None,
        log_after_iters=1,
        save_after_epochs=None,
        save_path='../saves/train',
        device='cpu',
        do_overfitting=False
    ):
        self.train_config = train_config

        if self.epoch == 0:
            self.optim = optim

        iter_per_epoch = len(train_loader)
        print("Iterations per epoch: {}".format(iter_per_epoch))

        # Exponentially filtered training loss
        train_loss_avg = 0

        # Path to save model and solver
        if save_path.split('/')[-1] == 'saves':
            save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        else:
            save_path = os.path.join(save_path)

        tensorboard_writer = SummaryWriter(
            os.path.join(tensorboard_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
            flush_secs=30)

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs * iter_per_epoch
        i_iter = 0

        print('Start training at epoch ' + str(self.epoch))
        t_start_training = time.time()

        # Training
        for i_epoch in range(num_epochs):
            self.epoch += 1
            if do_overfitting:
                print("Starting epoch {}  ".format(self.epoch), end='')
            else:
                print("Starting epoch {}".format(self.epoch))
            t_start_epoch = time.time()

            # Set model to train mode
            model.train()
            model.to(device)

            for i_iter_in_epoch, batch in enumerate(train_loader):
                t_start_iter = time.time()
                i_iter += 1

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                y_pred = model(x)

                # Compute loss
                train_loss = binary_cross_entropy(y_pred, y)

                # Backpropagate and update weights
                model.zero_grad()
                train_loss.backward()
                self.optim.step()

                # Logging
                smooth_window_train = 10
                train_loss_avg = (smooth_window_train - 1) / smooth_window_train * train_loss_avg + \
                                  1 / smooth_window_train * train_loss.item()

                if log_after_iters is not None and (i_iter % log_after_iters == 0):
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) +
                          "   Train loss: " + "{0:.6f}".format(train_loss.item()),
                          "   Avg train loss: " + "{0:.6f}".format(train_loss_avg) +
                          " - Time/iter: " + str(int((time.time() - t_start_iter) * 1000)) + "ms")

                self.append_history({
                    'train_loss': train_loss.item(),
                })

                tensorboard_writer.add_scalar('Train_loss', train_loss.item(), i_iter)

            if not do_overfitting:
                # Validate model
                print("\nValidate model after epoch " + str(self.epoch) + '/' + str(num_epochs))

                # Set model to evaluation mode
                model.eval()

                num_val_batches = 0
                val_loss = 0

                for batch in val_loader:
                    num_val_batches += 1

                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                    # Forward pass
                    y_pred = model(x)

                    # Compute loss
                    current_val_loss = binary_cross_entropy(y_pred, y)

                val_loss += current_val_loss.item()

                val_loss /= num_val_batches

                self.append_history({'val_loss': val_loss})

                print('Avg Train Loss: ' + "{0:.6f}".format(train_loss_avg) +
                      '   Val loss: ' + "{0:.6f}".format(val_loss) +
                      "   - " + str(int((time.time() - t_start_epoch) * 1000)) + "ms" +
                      "   time left: {}\n".format(time_left(t_start_training, n_iters, i_iter)))

            # Save model and solver
            if save_after_epochs is not None and (self.epoch % save_after_epochs == 0):
                os.makedirs(save_path, exist_ok=True)
                model.save(save_path + '/model' + str(self.epoch))
                self.training_time_s += time.time() - t_start_training
                self.save(save_path + '/solver' + str(self.epoch))
                model.to(device)

            # Stop if training time is over
            if max_train_time_s is not None and (time.time() - t_start_training > max_train_time_s):
                print("Training time is over.")
                break

        # Save model and solver after training
        os.makedirs(save_path, exist_ok=True)
        model.save(save_path + '/model' + str(self.epoch))
        self.training_time_s += time.time() - t_start_training
        self.save(save_path + '/solver' + str(self.epoch))

    def save(self, path):
        print('Saving solver... %s\n' % path)
        torch.save({
            'history'         : self.history,
            'epoch'           : self.epoch,
            'training_time_s' : self.training_time_s,
            'criterion'       : self.criterion,
            'optim_state_dict': self.optim.state_dict(),
            'train_config'    : self.train_config
        }, path)

    def load(self, path, device, only_history=False):

        checkpoint = torch.load(path, map_location=device)

        if not only_history:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.criterion = checkpoint['criterion']

        self.history = checkpoint['history']
        self.epoch = checkpoint['epoch']
        self.training_time_s = checkpoint['training_time_s']
        self.train_config = checkpoint['train_config']

    def append_history(self, hist_dict):
        for key in hist_dict:
            if key not in self.history:
                self.history[key] = [hist_dict[key]]
            else:
                self.history[key].append(hist_dict[key])
