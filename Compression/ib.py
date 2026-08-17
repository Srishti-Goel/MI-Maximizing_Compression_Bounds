import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Bounds():
    def __init__(self, name, cond_prob_model_class, **kwargs):
        self.name = name
        self.cond_prob_model = cond_prob_model_class
        self.cond_prob_model_kwargs = kwargs

class IB(nn.Module):
    def __init__(self, 
                 encoder,
                 cond_prob_model,
                 beta,
                 reg_weight,
                 reg_type=0,
                 optim_func=torch.optim.Adam,
                 **optim_kwargs
                ):
        '''
        Initializing the parameters with the following examples:
            encoder = IBCompressor(DATAPOINT_SIZE, LATENT_SIZE)
            cond_prob_model = ConditionalKDE()
            reg_weight = config.REGULARIZATION_WEIGHT
            reg_type = 
                0: std_dev of the encoder output
                1: IMNN regularization
            beta = config.BETA
        '''
        super().__init__()

        # The main models
        self.encoder = encoder
        self.cond_prob_model = cond_prob_model

        # Loss-config parameters
        self._beta = beta
        self._reg_weight = reg_weight

        # Lists to save the losses during training
        self.full_losses = []
        self.kl_losses = []
        self.log_prob_losses = []
        self.reg_losses = []

        self.reg_type = reg_type

        # Optimizer for training
        self.optim = optim_func(
            self.parameters(),
            **optim_kwargs
        )
        
    def plot_losses(self):
        '''
        Simple utility to plot the different losses as a function
        '''
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        axes[0].plot(self.full_losses)
        axes[0].set_title("Total Loss")
        axes[0].set_ylabel("Loss")
        axes[1].plot(self.log_prob_losses)
        axes[1].set_title("log_q_cond (should decrease)")
        axes[1].set_ylabel("log_q_cond")
        axes[2].plot(self.kl_losses)
        axes[2].set_title("KL Divergence")
        axes[2].set_ylabel("KL")
        axes[3].plot(self.reg_losses)
        axes[3].set_title("Regularization Loss")
        axes[3].set_ylabel("Reg. Loss")
        plt.tight_layout()
        plt.show()
    
    def compute_loss(self, x, y):
        '''
        Compute the loss given the uncompressed data x, and the true parameters y
        '''
        # Forward step of the encoder
        mean, logvar, z = self.encoder(x)

        if torch.isnan(mean).any() or torch.isnan(logvar).any() or torch.isnan(z).any():
            print("Encoder becomes nans first")
            print("Mean:", torch.isnan(mean))
            print("Logvar", torch.isnan(logvar))
            print("z:", torch.isnan(z))
            return

        # Forward step of the conditional probability model
        z_mean_train = z.mean()
        z_std_train = z.std()
        z_normalized = (z - z_mean_train) / z_std_train
        log_q_cond = self.cond_prob_model(y, z_normalized)
        if torch.isnan(log_q_cond).any():
            print("cond prob model becomes nans first")
            return

        # KL divergence (term added in this model)
        kl = 0.5 * torch.mean(mean ** 2 + logvar.exp() - torch.log(logvar.exp()))
        if torch.isnan(kl):
            print("KL becomes nans first")
            return

        # Regularization term to prevent the model from just scaling up its output to apparently increase information content
        cov = torch.std(z)**2
        lamb = (cov - 1) + ((1 / (cov + 1e-6)) - 1)
        reg_loss = self._reg_weight * (
            (
                (1 - self.reg_type) * cov
            ) + (
                self.reg_type * lamb**2 / (lamb - torch.exp(-2*lamb) + 1e-6)
            ))
        if torch.isnan(reg_loss):
            print("Surprise! Reg loss becomes nan first!!")
            return

        # Putting it all together
        loss = -(log_q_cond).mean() + (self._beta * kl) + reg_loss
        
        return loss, log_q_cond.mean(), kl, reg_loss
    
    def train(self, x_train, y_train, nepochs, batch_size=None, print_losses=True):
        '''
        Train the model for another nepochs based on the saved during initialization
        '''

        # Training configuration is set during the initialization because these are ideally different between different experiments

        # Ensure inputs are tensors
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train)

        n_samples = x_train.shape[0]
        use_minibatch = (batch_size is not None) and (batch_size < n_samples)

        if use_minibatch:
            dataset = TensorDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(nepochs):

            if use_minibatch:
                epoch_full = 0.0
                epoch_log_q = 0.0
                epoch_kl = 0.0
                epoch_reg = 0.0
                nbatches = 0

                for xb, yb in loader:
                    self.optim.zero_grad()
                    loss, log_q_cond, kl, reg_loss = self.compute_loss(xb, yb)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optim.step()

                    epoch_full += loss.item()
                    epoch_log_q += log_q_cond.item()
                    epoch_kl += kl.item()
                    epoch_reg += reg_loss.item()
                    nbatches += 1

                    self.full_losses.append(loss.item())
                    self.log_prob_losses.append(log_q_cond.item())
                    self.kl_losses.append(kl.item())
                    self.reg_losses.append(reg_loss.item())

                # Save epoch-averaged losses
                avg_full = epoch_full / nbatches
                avg_log_q = epoch_log_q / nbatches
                avg_kl = epoch_kl / nbatches
                avg_reg = epoch_reg / nbatches


                if print_losses and (epoch == 0 or (epoch+1) % 10 == 0):
                    print(f"Epoch {epoch+1}: Loss={avg_full:.4f}, log_q={avg_log_q:.4f},, reg. loss={avg_reg:.4f}, beta * KL={avg_kl:.4f}")
            else:
                self.optim.zero_grad()
                # Use the self function to calculate the loss 
                loss, log_q_cond, kl, reg_loss = self.compute_loss(x_train, y_train)

                # Backward-propogation
                loss.backward()
                # Clipping gradients to prevent model from collapsing
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optim.step()

                # Appending to loss-saving lists
                self.full_losses.append(loss.item())
                self.log_prob_losses.append(log_q_cond.item())
                self.kl_losses.append(kl.item())
                self.reg_losses.append(reg_loss.item())

                if print_losses and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={loss.item():.4f}, log_q={log_q_cond.item():.4f},, reg. loss={reg_loss.item():.4f}, beta * KL={kl.item():.4f}")
        
        if print_losses:
            self.plot_losses()
    
    def evaluate_beta(self, test_sims_data):
        pass
    
    def forward(self, x):
        mean, logvar, z = self.encoder(x)

        return mean