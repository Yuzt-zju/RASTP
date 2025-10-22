from typing import Any, Dict, Optional, Union

import torch
import transformers
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.aggregation import BaseAggregator

from src.components.eval_metrics import Evaluator
from src.utils.pylogger import RankedLogger

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


class BaseModule(LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, transformers.PreTrainedModel],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: torch.nn.Module,
        evaluator: Evaluator,
        training_loop_function: callable = None,
    ) -> None:
        """
        Args:
            model: The model to train.
            optimizer: The optimizer to use for the model.
            scheduler: The scheduler to use for the model.
            loss_function: The loss function to use for the model.
            evaluator: The evaluator to use for the model.
            training_loop_function: The training loop function to use for the model, in case it is different than the default one.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.evaluator = evaluator
        self.training_loop_function = training_loop_function
        # We use setters to set the prediction key and name.
        self._prediction_key_name = None
        self._prediction_name = None

        if self.training_loop_function is not None:
            self.automatic_optimization = False

        if self.evaluator:  # For inference, evaluator is not set.
            for metric_name, metric_object in self.evaluator.metrics.items():
                setattr(self, metric_name, metric_object)

            # for averaging loss across batches
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
            self.test_loss = MeanMetric()

    @property
    def prediction_key_name(self) -> Optional[str]:
        return self._prediction_key_name

    @prediction_key_name.setter
    def prediction_key_name(self, value: str) -> None:
        command_line_logger.debug(f"Setting prediction_key_name to {value}")
        self._prediction_key_name = value

    @property
    def prediction_name(self) -> Optional[str]:
        return self._prediction_name

    @prediction_name.setter
    def prediction_name(self, value: str) -> None:
        command_line_logger.debug(f"Setting prediction_name to {value}")
        self._prediction_name = value

    def forward(
        self,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def model_step(
        self,
        model_input: Any,
        label_data: Optional[Any] = None,
    ):
        raise NotImplementedError(
            "Inherit from this class and implement the model_step method."
        )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.evaluator.reset()
        self.train_loss.reset()
        self.test_loss.reset()

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        self.val_loss.reset()
        self.evaluator.reset()

    def on_test_epoch_start(self):
        self.test_loss.reset()
        self.evaluator.reset()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.log("val/loss", self.val_loss, sync_dist=False, prog_bar=True, logger=True)
        self.log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self.log(
            "test/loss", self.test_loss, sync_dist=False, prog_bar=True, logger=True
        )
        self.log_metrics("test")

    def on_exception(self, exception):
        self.trainer.should_stop = True  # stop all workers
        self.trainer.logger.finalize(status="failure")

    def log_metrics(
        self,
        prefix: str,
        on_step=False,
        on_epoch=True,
        # We use sync_dist=False by default because, if using retrieval metrics, those are already synchronized. Change if using
        # different metrics than the default ones.
        sync_dist=False,
        logger=True,
        prog_bar=False,
        call_compute=False,
    ) -> Dict[str, Any]:

        metrics_dict = {
            f"{prefix}/{metric_name}": metric_object.compute()
            if call_compute
            else metric_object
            for metric_name, metric_object in self.evaluator.metrics.items()
        }

        self.log_dict(
            metrics_dict,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            logger=logger,
            prog_bar=prog_bar,
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def eval_step(self, batch: Any, loss_to_aggregate: BaseAggregator):
        raise NotImplementedError("eval_step method must be implemented.")

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.val_loss)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.test_loss)
