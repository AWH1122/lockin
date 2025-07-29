from abc import ABC
from functools import partial

import lightning as L
import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError
from torchmetrics.wrappers import MultioutputWrapper

from fxguard.data.timeseries import TimeSeries, TimeSeriesDataset

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def collate_dict(batch: list[dict]):
    out: dict[str, torch.Tensor | list | dict] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]
        if torch.is_tensor(v0):
            out[k] = torch.stack(vals)
        elif isinstance(v0, np.ndarray):
            out[k] = torch.as_tensor(np.stack(vals))
        elif isinstance(v0, (float, int)):
            out[k] = torch.as_tensor(vals)
        else:
            out[k] = vals[0]  # weird I need to do this
    return out

class BaseLightning(L.LightningModule, ABC):
    def __init__(
        self,
        input_length: int,
        h: int | list[int] = 1,
        n_epochs: int = 100,
        batch_size: int = 32,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        lr_scheduler_cls: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_length = input_length
        self.h = h if isinstance(h, list) else [h]
        self.H = max(self.h)
        trainer_kwargs = trainer_kwargs or {}
        self.trainer = partial(Trainer, max_epochs=n_epochs, **trainer_kwargs)
        self.criterion = MeanSquaredError()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_cls is not None:
            return [optimizer], [self.lr_scheduler_cls(optimizer, **self.lr_scheduler_kwargs)]
        return optimizer

    def training_step(self, batch: Batch):
        x = batch['future_target']  # should be (batch_size,n_horizons,1)
        xhat = self(batch)
        loss = self.criterion(xhat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Batch):
        x = batch['future_target']  # should be (batch_size,n_horizons,1)
        xhat = self(batch)
        loss = self.criterion(xhat, x)
        self.log('val_loss', loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._dataloader_kwargs, collate_fn=collate_dict)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, **self._dataloader_kwargs, collate_fn=collate_dict)
        return []

    # def setup(self, stage):    #DON'T REMOVE!
    #     if stage == 'fit':
    #         sample = next(iter(self.train_dataloader()))
    #         input_dim = self.input_length
    #         output_dim = self.H
    #         past_cov_dim = sample['past_covariates'].shape[-1]
    #         future_cov_dim = sample['future_covariates'].shape[-1]
    #         static_cov_dim = sample['static_covariates'].shape[-1]

    def fit(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        val_series=None,
        val_past_covariates=None,
        val_future_covariates=None,
        epochs=None,
        dataloader_kwargs=None,
    ):
        self.trainer: Trainer
        self._dataloader_kwargs = dataloader_kwargs or {}
        if 'shuffle' not in self._dataloader_kwargs:
            self._dataloader_kwargs['shuffle'] = True
        if 'batch_size' not in self._dataloader_kwargs:
            self._dataloader_kwargs['batch_size'] = self.hparams.batch_size
        if epochs is not None:
            self.trainer = self.trainer(max_epochs=epochs)
        else:
            self.trainer = self.trainer()
        self.train_dataset = TimeSeriesDataset(
            series,
            input_length=self.input_length,
            h=self.h,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        if val_series is not None:
            self.val_dataset = TimeSeriesDataset(
                val_series,
                input_length=self.input_length,
                h=self.h,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
            )
        else:
            self.val_dataset = None
        self.trainer.fit(self)
        # return self

    def predict(
        self,
        series: TimeSeries | list[TimeSeries],
        past_covariates: TimeSeries | list[TimeSeries] | None = None,
        future_covariates: TimeSeries | list[TimeSeries] | None = None,
        batch_size: int | None = None,
        dataloader_kwargs: dict | None = None,
    ):
        trainer: Trainer
        if batch_size is not None:
            trainer = self.trainer(batch_size=batch_size)
        else:
            trainer = self.trainer()
        last_chunks = [ts[: -self.input_length] for ts in series]
        predict_dataset = TimeSeriesDataset(last_chunks, past_covariates, future_covariates)
        return trainer.predict(
            self,
            dataloaders=self.predict_dataloader(predict_dataset, dataloader_kwargs),  # need to remove dataloader
        )  # the shape here will have length equal to num input series

    def historical_forecast(self, series, past_covariates=None, future_covariates=None, stride: int = 1):
        predictions = []
        dataset = TimeSeriesDataset(
            series,
            input_length=self.input_length,
            h=self.h,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            stride=stride,
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_dict)
        for batch in dataloader:
            predictions.extend(self(batch).cpu().flatten().tolist())
        return predictions
