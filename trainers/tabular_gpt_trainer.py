import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from min_gpt_utils import CfgNode

from trainers.min_gpt_trainer import Trainer

class TrainerWithVal(Trainer):
    def __init__(self, config, model, train_dataset, val_dataset, save_weights_path=None):
        super().__init__(config, model, train_dataset)
        self.val_dataset = val_dataset
        self.save_weights_path = save_weights_path  # string where to save weights i.e. "best_model_path.pth"

    def run(self, log_level=0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model, config = self.model, self.config

        # set up the optimizer
        self.optimizer = model.configure_optimizers(config)

        # set up the dataloader
        # note: here we don't need a sampler really, not sure why Karpathy
        # did that except it kind of works in his specific case
        # but ours we can just use the data and randomly sample batches from
        # it instead of go through a 10^10 replaced dataset...?
        train_loader = DataLoader(
            self.train_dataset,
            # sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=True,  # just shuffle the data
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # set up the validation dataloader
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=True,  # why not shuffle here too
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            # batch = [t.to(self.device) for t in batch]
            x, y = batch
            # x = {key: torch.stack([t.to(device).long() if 'embed' in key else t.to(device) for t in values]) for key, values in x.items()}
            # x = {key: t.to(device).long() if 'embed' in key else t.to(device) for key, t in x.items()}
            x = x.to(device)
            y = y.to(device)

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # Validation
            if self.iter_num % config.val_interval == 0:
                if log_level:
                    tnow = time.time()
                    self.iter_dt = tnow - self.iter_time
                    self.iter_time = tnow
                    print(f"iter_dt {self.iter_dt * 1000:.2f}ms; iter {self.iter_num}: train loss {self.loss.item():.5f}")

                model.eval()  # Set the model to evaluation mode
                # with torch.no_grad():
                #     val_losses = []
                #     for val_batch in val_loader:
                #         # val_batch = [t.to(self.device) for t in val_batch]
                #         val_x, val_y = val_batch
                #         # val_x = {key: t.to(device).long() if 'embed' in key else t.to(device) for key, t in val_x.items()}
                #         val_x = val_x.to(device)
                #         val_y = val_y.to(device)
                #         val_logits, val_loss = model(val_x, val_y)
                #         val_losses.append(val_loss.item())
                #     average_val_loss = sum(val_losses) / len(val_losses)

                with torch.no_grad():
                    val_losses = []
                    val_correct = 0
                    val_total = 0

                    for val_batch in val_loader:
                        val_x, val_y = val_batch
                        val_x = val_x.to(device)
                        val_y = val_y.to(device)

                        val_logits, val_loss = model(val_x, val_y)
                        val_losses.append(val_loss.item())

                        # Convert logits to probabilities using sigmoid
                        print("val logits shape: ", val_logits.shape)
                        print("val y shape: ", val_y.shape)
                        val_probs = torch.sigmoid(val_logits[:,-1,:])

                        # Threshold probabilities to get binary predictions
                        val_preds = (val_probs[:, -1] >= 0.5).float()

                        # Calculate accuracy
                        val_correct += (val_preds == val_y).sum().item()
                        val_total += val_y.size(0)

                    average_val_loss = sum(val_losses) / len(val_losses)
                    val_accuracy = val_correct / val_total

                    if log_level:
                        print("avg val loss: ", average_val_loss)
                        print("avg val acc: ", val_accuracy)

                # Update the best validation loss and check for improvement
                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    epochs_without_improvement = 0
                    # Save the model weights
                    if self.save_weights_path:
                        torch.save(model.state_dict(), self.save_weights_path)
                else:
                    epochs_without_improvement += 1

                # Terminate if validation loss doesn't improve for a certain number of epochs
                if epochs_without_improvement >= config.patience:
                    break

                model.train()  # set model back to train

            # iter termination conditions
            self.iter_num += 1

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
