import os
import csv
import torch

from tqdm import tqdm
#from model.fno import D
import wandb

class Trainer:
    def __init__(
        self,
        model,
        mask,
        config,
        data_processor=None,
    ):
        self.model = model
        self.n_epochs = config.opt.epochs
        self.verbose = config.verbose
        #self.use_distributed = config.use_distributed
        self.device = config.device
        #self.wandb_log = config.wandb_log
        self.mask = mask
        self.min_epoch = config.min_epoch
        self.variable = config.data.variable
        # self.lifting_channels = config.fno.lifting_channels
        self.config = config       
        

    def train(
        self,
        train_data_loader,
        val_data_loader,
        mse,
        optimizer,
        scheduler,
        training_loss=None,
        eval_loss=None,
    ):
        """Trains the given model on the given datasets.
        params:
        data_loader: torch.utils.data.DataLoader
             dataloader
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        """

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(self.n_epochs):

            print(f"Starting epoch {epoch}:")
            pbar = tqdm(train_data_loader)
            self.model.train()
            train_loss = 0.0
            total_samples = 0
            for i, (data, output) in enumerate(pbar):
                
                # Input and output to device
                data = data.to(self.device)
                output = output.to(self.device)
                mask = self.mask.to(self.device)

                # masking actual and predicted output
                generated_data = self.model(data)
                output = output*mask
                generated_data = generated_data*mask

                # loss
                loss = mse(generated_data, output)
                train_loss += loss.item()*len(data)
                total_samples += len(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss /= total_samples
            train_losses.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                test_loss = 0.0
                total_samples = 0
                pbar = tqdm(val_data_loader)

                for i, (test_input, test_output) in enumerate(pbar):

                    data_test = test_input.to(self.device)
                    test_output = test_output.to(self.device)
                    output_test = self.model(data_test)
                    test_output = test_output*mask
                    output_test = output_test*mask
                    loss_test = mse(output_test, test_output)
                    test_loss += loss_test.item()*len(data_test)
                    total_samples += len(data_test)

                test_loss /= total_samples
                val_losses.append(test_loss)
                scheduler.step(test_loss)
                print(f'lr: {optimizer.param_groups[0]['lr']}')
                print(f'Device: {self.device}, Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {test_loss}')
                logs = {"train loss": train_loss, "val loss": test_loss}

            if test_loss < best_val_loss and epoch > self.min_epoch:
                best_val_loss = test_loss
                filename = (
                    f"BOB_model_mean_zos_input_oc_{self.config.data.variable}_atm_{self.config.data.atm_variable}_output_{self.config.data.out_variable}_patch_{self.config.data.patch_size}_"
                    f"emd_dim_{self.config.data.emd_dim}_afno_layers_{self.config.afno2d.n_blocks}_blocks_{self.config.afno2d.num_blocks}_hd_{self.config.afno2d.hidden_size}"
                    f"mlp_in_{self.config.mlp.in_features}_hd_{self.config.mlp.hidden_features}_lr_{self.config.opt.lr}.pth"
                )
                torch.save(self.model.state_dict(), f"saved_models/{filename}")
                print(f"Epoch {epoch}: Model saved.")
                best_epoch = epoch

            if self.config.wb:
                wandb.log(logs, step=epoch+1)

        print('Best validation loss', best_val_loss)
        print('epoch at which model is stored', best_epoch)
        # After all epochs, save the losses to a CSV file
        os.makedirs("loss", exist_ok=True)  # Ensure the 'loss' directory exists
        filename_csv = ("loss/" + filename + ".csv")
        with open(filename_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])  # Header
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                writer.writerow([epoch, train_loss, val_loss])

        print("Training and validation losses saved to 'loss/train_val_losses.csv'")


