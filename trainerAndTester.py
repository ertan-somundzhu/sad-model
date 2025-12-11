
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchsummary import summary

from typing import Optional, List, Callable


class trainAndTestModel:

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: str,
        optimizer: torch.optim,
        criterion: torch.nn.modules.loss,
        metrics: Optional[dict[tuple[Callable]]] = None,
        n_epochs: int = 30,
        # following parameters are included primarily
        # due to the usage of bcewithlogitsloss
        # as even though everithing is fine when computing the loss itself
        # an activation function is necessary when calculating metrics like f1, recall, precision, etc.
        # and yes, i know it looks somewhat clunky
        output_is_raw_logits: bool = False,
        final_activation_function_for_raw_logits: Optional[torch.nn.modules.activation] = None 
    ):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.metrics = metrics

        self.output_is_raw_logits = output_is_raw_logits
        self.final_activation_function_for_raw_logits = final_activation_function_for_raw_logits

        self.have_metrics = False if self.metrics is None else True

    def _compute_additional_losses(self, y_true, y_pred, loss_func):

        with torch.no_grad():
            
            assert y_true.shape == y_pred.shape

            y_true_bin = y_true.bool()
     
            y_pred_bin = (y_pred >= 0.5)

            loss = loss_func(y_true_bin, y_pred_bin)

            return loss

    def train(self):

        scaler = torch.amp.GradScaler(self.device)


        history = {"loss": {"train": [], "val": []}}

        if self.have_metrics:

            for metric in self.metrics.keys():

                history[metric] = {"train": [], "val": []}


        for epoch in tqdm(range(self.n_epochs)):

            if self.have_metrics:

                additional_losses = {}

                for metric in self.metrics.keys():
                    additional_losses[metric] = {"train": 0, "val": 0}

            self.model.train()
            running_loss = 0.0

            count = 0

            for inputs, target in self.train_loader:

                inputs = inputs.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(self.device):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, target)

                if self.have_metrics:
                    for metric, loss_func in self.metrics.items():

                        additional_losses[metric]["train"] += self._compute_additional_losses(
                            y_true=target,
                            y_pred=self.final_activation_function_for_raw_logits(outputs) if self.output_is_raw_logits else outputs,
                            loss_func=loss_func
                        )


                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()


                running_loss += loss.item()
                count += 1

            train_loss = running_loss / count

            if self.have_metrics:
                for metric in self.metrics.keys():

                    history[metric]["train"].append(additional_losses[metric]["train"] / count)


            self.model.eval()
            with torch.no_grad():

                running_loss = 0
                count = 0

                for inputs, target in self.val_loader:

                    inputs = inputs.to(self.device)
                    target = target.to(self.device)

                    with torch.amp.autocast(self.device):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, target)

                    running_loss += loss.item()
                    count += 1

                    if self.have_metrics:

                        for metric, loss_func in self.metrics.items():

                            additional_losses[metric]["val"] += self._compute_additional_losses(
                                y_true=target,
                                y_pred=self.final_activation_function_for_raw_logits(outputs) if self.output_is_raw_logits else outputs,
                                loss_func=loss_func
                            )


                val_loss = running_loss / count

                if self.have_metrics:
                    for metric in self.metrics.keys():

                        history[metric]["val"].append(additional_losses[metric]["val"] / count)

            history["loss"]["train"].append(train_loss)
            history["loss"]["val"].append(val_loss)



        return self.model, history

    def test(self, print_test_metrics: bool = True, return_test_metrics: bool = False):

        self.model.eval()
        with torch.no_grad():

            if self.have_metrics:

                test_metrics = {metric: 0 for metric in self.metrics.keys()}

                test_metrics["loss"] = 0

            running_loss = 0
            count = 0

            for inputs, target in self.test_loader:

                inputs = inputs.to(self.device)
                target = target.to(self.device)

                with torch.amp.autocast(self.device):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, target)

                running_loss += loss.item()
                count += 1


                if self.have_metrics:

                    for metric, loss_func in self.metrics.items():

                         test_metrics[metric] += self._compute_additional_losses(
                            y_true=target,
                            y_pred=self.final_activation_function_for_raw_logits(outputs) if self.output_is_raw_logits else outputs,
                            loss_func=loss_func
                        )
                         
            for metric, score in test_metrics.items():

                test_metrics[metric] = score / count

            test_metrics["loss"] = running_loss / count


        if print_test_metrics:

            for metric, score in test_metrics.items():

                print(f"{metric.capitalize()}: {score}")


        if return_test_metrics:
            return test_metrics
