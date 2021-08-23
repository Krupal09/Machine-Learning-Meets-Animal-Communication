"""
Adapted from orcaspot's module: trainer.py
By R.X.Cheng
"""

import os
import copy
import math
import time
import operator
import numpy as np
import utils.metrics as m

import torch
import torch.nn as nn


from typing import Union
from utils.logging import Logger
from tensorboardX import SummaryWriter
from utils.summary import prepare_img
from delve import CheckLayerSat

from utils.checkpoints import CheckpointHandler
from utils.early_stopping import EarlyStoppingCriterion

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            logger: Logger,
            prefix: str="",
            checkpoint_dir: Union[str, None] = None,
            summary_dir: Union[str, None] = None,
            n_summaries: int=4, #
            input_shape: tuple=None,
            start_scratch: bool=False,
            #model_name: str="model",
    ):
        """
        Class which implements network training, validation and testing as well as writing checkpoints, logs, summaries, and saving the final model.

        :param Union[str, None] checkpoint_dir: the type is either str or None (default: None)
        :param int n_summaries: number of images as samples at different phases to visualize on tensorboard
        """
        #self.model_name=model_name
        self.model = model
        self.logger = logger
        self.prefix = prefix

        self.logger.info("Init summary writer")

        if summary_dir is not None:
            run_name = prefix + "_" if prefix != "" else ""
            run_name += "{time}-{host}".format(
                time=time.strftime("%y-%m-%d-%H-%M", time.localtime()),
                host=os.uname()[1],
            )
            self.summary_dir = os.path.join(summary_dir, run_name)

        self.n_summaries = n_summaries
        self.writer = SummaryWriter(summary_dir)

        if input_shape is not None:
            dummy_input = torch.rand(input_shape)
            self.logger.info("Writing graph to summary")
            self.writer.add_graph(self.model, dummy_input)

        if checkpoint_dir is not None:
            self.cp = CheckpointHandler(
                checkpoint_dir, prefix=prefix, logger=self.logger
            )
        else:
            self.cp = None

        self.start_scratch = start_scratch

    def fit(
            self,
            train_dataloader,
            val_dataloader,
            train_ds,
            val_ds,
            loss_fn,
            optimizer,
            n_epochs,
            val_interval,
            patience_early_stopping,
            device,
            metrics: Union[list, dict] = [],
            val_metric: Union[int, str] = "loss",
            val_metric_mode: str = "min",
            start_epoch=0,
    ):
        """
        train and validate the networks

        :param int n_epochs: max_train_epochs (default=500)
        :param int val_interval: run validation every val_interval number of epoch (ARGS.patience_early_stopping)
        :param int patience_early_stopping: after (patience_early_stopping/val_interval) number of epochs without improvement, terminate training
        """

        self.logger.info("Init model on device '{}'".format(device))
        self.model = self.model.to(device)

        # initalize delve
        self.tracker = CheckLayerSat(self.summary_dir, save_to="plotcsv", modules=self.model, device=device)

        best_model = copy.deepcopy(self.model.state_dict())
        best_metric = 0.0 if val_metric_mode == "max" else float("inf")

        # as we don't validate after each epoch but at val_interval,
        # we update the patience_stopping accordingly to how many times of validation
        patience_stopping = math.ceil(patience_early_stopping / val_interval)
        patience_stopping = int(max(1, patience_stopping))
        early_stopping = EarlyStoppingCriterion(
            mode=val_metric_mode, patience=patience_stopping
        )

        if not self.start_scratch and self.cp is not None:
            checkpoint = self.cp.read_latest()
            if checkpoint is not None:
                try:
                    try:
                        self.model.load_state_dict(checkpoint["modelState"])
                    except RuntimeError as e:
                        self.logger.error(
                            "Failed to restore checkpoint: "
                            "Checkpoint has different parameters"
                        )
                        self.logger.error(e)
                        raise SystemExit

                    optimizer.load_state_dict(checkpoint["trainState"]["optState"])
                    start_epoch = checkpoint["trainState"]["epoch"] + 1
                    best_metric = checkpoint["trainState"]["best_metric"]
                    best_model = checkpoint["trainState"]["best_model"]
                    early_stopping.load_state_dict(
                        checkpoint["trainState"]["earlyStopping"]
                    )
                    #scheduler.load_state_dict(checkpoint["trainState"]["scheduler"])
                    self.logger.info("Resuming with epoch {}".format(start_epoch))
                except KeyError:
                    self.logger.error("Failed to restore checkpoint")
                    raise

        since = time.time()

        self.logger.info("Start training model " + self.prefix)

        try:
            if val_metric_mode == "min":
                val_comp = operator.lt # to run standard operator as function
            else:
                val_comp = operator.gt
            for epoch in range(start_epoch, n_epochs):
                self.train(
                    epoch, train_dataloader, train_ds, loss_fn, optimizer, device
                )

                if epoch % val_interval == 0 or epoch == n_epochs - 1:
                    # first, get val_loss for further comparison
                    val_loss = self.validate(
                        epoch, val_dataloader, val_ds, loss_fn, device, phase="val"
                    )
                    if val_metric == "loss":
                        val_result = val_loss
                        # add metrics for delve to keep track of
                        self.tracker.add_scalar("loss", val_loss)
                        # add saturation to the mix
                        self.tracker.add_saturations()
                    else:
                        val_result = metrics[val_metric].get()

                    # compare to see if improvement occurs
                    if val_comp(val_result, best_metric):
                        best_metric = val_result # update best_metric with the loss (smaller than previous)
                        best_model = copy.deepcopy(self.model.state_dict())
                        """previously, deadlock occurred, which seemed to be related to cp. comment self.cp.write() to see if freezing goes away."""
                        # write checkpoint
                        self.cp.write(
                            {
                                "modelState": self.model.state_dict(),
                                "trainState": {
                                    "epoch": epoch,
                                    "best_metric": best_metric,
                                    "best_model": best_model,
                                    "optState": optimizer.state_dict(),
                                    "earlyStopping": early_stopping.state_dict(),
                                },
                            }
                        )

                    # test if the number of accumulated no-improvement epochs is bigger than patience
                    if early_stopping.step(val_result):
                        self.logger.info(
                            "No improvement over the last {} epochs. Training is stopped.".format(patience_early_stopping)
                        )
                        break
        except Exception:
            import traceback
            self.logger.warning(traceback.format_exc())
            self.logger.warning("Aborting...")
            self.logger.close()
            raise SystemExit

        # option here: load the best model to run test on test_dataset and log the final metric (along side best metric)
        # for ae, only split: train and validate dataset, without test_dataset

        time_elapsed = time.time() - since
        self.logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        self.logger.info("Best val metric: {:4f}".format(best_metric))

        # close delve tracker
        self.tracker.close()

        return self.model

    def train(self, epoch, train_dataloader, train_ds, loss_fn, optimizer, device):
        """
        Training of one epoch on training data, loss function, optimizer, and respective metrics
        """
        self.logger.debug("train|{}|start".format(epoch))

        self.model.train()

        epoch_start = time.time()
        start_data_loading = epoch_start
        data_loading_time = m.Sum(torch.device("cpu"))

        train_running_loss = 0.0
        for i, (train_specs, label) in enumerate(train_dataloader):
            train_specs = train_specs.to(device)
            call_label = None

            if "call" in label:
                call_label = label["call"].to(device, non_blocking=True, dtype=torch.int64)

            data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))
            optimizer.zero_grad()

            # compute reconstructions
            outputs = self.model(train_specs)

            # compute training reconstruction loss
            loss = loss_fn(outputs, train_specs)

            # compute accumulated gradients
            loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            # the value of total cost averaged across all training examples of the current batch
            # loss.item()*data.size(0): total loss of the current batch (not averaged).
            train_running_loss += loss.item() * train_specs.size(0)

            prediction = None

            if i % 2 == 0:
                self.write_summaries(
                    features=train_specs,
                    labels=call_label,
                    prediction=prediction,
                    file_names=label["file_name"],
                    epoch=epoch,
                    phase="train",
                )
            start_data_loading = time.time()

            # compute the epoch training loss
        train_epoch_loss = train_running_loss / len(train_ds)

        self.write_scalar_summaries_logs(
            loss=train_epoch_loss,
            #metrics=metrics,
            lr=optimizer.param_groups[0]["lr"],
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase="train",
        )

        self.writer.flush()

        return train_epoch_loss

    def validate(self, epoch, val_dataloader, val_ds, loss_fn, device, phase="val"):
        self.logger.debug("{}|{}|start".format(phase, epoch))
        self.model.eval()

        val_running_loss = 0.0
        with torch.no_grad():
            epoch_start = time.time()
            start_data_loading = epoch_start
            data_loading_time = m.Sum(torch.device("cpu"))

            for i, (val_specs, label) in enumerate(val_dataloader):
                val_specs = val_specs.to(device)
                if "call" in label:
                    call_label = label["call"].to(device, non_blocking=True, dtype=torch.int64)

                data_loading_time.update(
                    torch.Tensor([(time.time() - start_data_loading)])
                )

                #if counter == 0:
                    # tb = SummaryWriter()
                    # grid = make_grid(val_specs)
                    #tb_writer.add_images("Original", val_specs, epoch)
                    # tb.close()

                outputs = self.model(val_specs) # logits

                #if counter == 0:
                    # tb = SummaryWriter()
                    # grid = make_grid(outputs)
                    #tb_writer.add_images("Reconstructed", outputs, epoch)
                    # tb.close()

                loss = loss_fn(outputs, val_specs)

                val_running_loss += loss.item() * val_specs.size(0)

                prediction = None

                if i % 2 == 0:
                    self.write_summaries(
                        features=val_specs,
                        labels=call_label,
                        prediction=prediction,
                        file_names=label["file_name"],
                        epoch=epoch,
                        phase=phase,
                    )
                start_data_loading = time.time()

            val_epoch_loss = val_running_loss / len(val_ds)

            self.write_scalar_summaries_logs(
                loss=val_epoch_loss,
                #metrics=metrics,
                epoch_time=time.time() - epoch_start,
                data_loading_time=data_loading_time.get(),
                epoch=epoch,
                phase=phase,
            )

            self.writer.flush()

            return val_epoch_loss

    def write_summaries(
            self,
            features,
            labels=None,
            prediction=None,
            file_names=None,
            epoch=None,
            phase="train",
    ):
        """Writes image summary per partition (spectrograms and the corresponding predictions)"""

        with torch.no_grad():
            self.write_img_summaries(
                features,
                labels=labels,
                prediction=prediction,
                file_names=file_names,
                epoch=epoch + 1,
                phase=phase,
            )


    def write_img_summaries(
            self,
            features,
            labels=None,
            prediction=None,
            file_names=None,
            epoch=None,
            phase="train",
    ):
        """
        Writes image summary per partition with respect to the prediction output (true predictions - true positive/negative, false
        predictions - false positive/negative)
        """

        with torch.no_grad():
            if file_names is not None:
                if isinstance(file_names, torch.Tensor):
                    file_names = file_names.cpu().numpy()
                elif isinstance(file_names, list):
                    file_names = np.asarray(file_names)
            if labels is not None and prediction is not None:
                features = features.cpu()
                labels = labels.cpu()
                prediction = prediction.cpu()
                for label in torch.unique(labels):
                    label = label.item()
                    l_i = torch.eq(labels, label)

                    t_i = torch.eq(prediction, label) * l_i
                    name_t = "true_{}".format("positive" if label else "negative")
                    try:
                        self.writer.add_image(
                            tag=phase + "/" + name_t,
                            img_tensor=prepare_img(
                                features[t_i],
                                num_images=self.n_summaries,
                                file_names=file_names[t_i.numpy() == 1],
                            ),
                            global_step=epoch,
                        )
                    except ValueError:
                        pass

                    f_i = torch.ne(prediction, label) * l_i
                    name_f = "false_{}".format("negative" if label else "positive")
                    try:
                        self.writer.add_image(
                            tag=phase + "/" + name_f,
                            img_tensor=prepare_img(
                                features[f_i],
                                num_images=self.n_summaries,
                                file_names=file_names[f_i.numpy() == 1],
                            ),
                            global_step=epoch,
                        )
                    except ValueError:
                        pass
            else:
                self.writer.add_image(
                    tag=phase + "/input",
                    img_tensor=prepare_img(
                        features, num_images=self.n_summaries, file_names=file_names
                    ),
                    global_step=epoch,
                )

    """
    Writes scalar summary per partition including loss, confusion matrix, accuracy, recall, f1-score, true positive rate,
    false positive rate, precision, data_loading_time, epoch time
    """
    def write_scalar_summaries_logs(
            self,
            loss: float,
            metrics: Union[list, dict] = [],
            lr: float = None,
            epoch_time: float = None,
            data_loading_time: float = None,
            epoch=None,
            phase="train",
    ):
        with torch.no_grad():
            log_str = phase
            if epoch is not None:
                log_str += "|{}".format(epoch)
            self.writer.add_scalar(phase + "/epoch_loss", loss, epoch)
            log_str += "|loss:{:0.3f}".format(loss)
            if isinstance(metrics, dict):
                for name, metric in metrics.items():
                    self.writer.add_scalar(phase + "/" + name, metric.get(), epoch)
                    log_str += "|{}:{:0.3f}".format(name, metric.get())
            else:
                for i, metric in enumerate(metrics):
                    self.writer.add_scalar(
                        phase + "/metric_" + str(i), metric.get(), epoch
                    )
                    log_str += "|m_{}:{:0.3f}".format(i, metric.get())
            if lr is not None:
                self.writer.add_scalar("lr", lr, epoch)
                log_str += "|lr:{:0.2e}".format(lr)
            if epoch_time is not None:
                self.writer.add_scalar(phase + "/time", epoch_time, epoch)
                log_str += "|t:{:0.1f}".format(epoch_time)
            if data_loading_time is not None:
                self.writer.add_scalar(
                    phase + "/data_loading_time", data_loading_time, epoch
                )
            self.logger.info(log_str)