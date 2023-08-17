#!/usr/bin/env/python3
import os
import sys
from matplotlib.pyplot import clabel
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
import h5py

from pathlib import Path
import time
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from gwdataset_torch import GW_SE_Dataset, WaveformDatasetTorch


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, targets):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Denoise
        mix_w = self.hparams.Encoder(mix.unsqueeze(-1)) # mlp encoder
        mix_w = torch.tanh(mix_w).permute(0, 2, 1)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = est_mask * mix_w

        est_source = torch.cat(
            [self.hparams.Decoder(sep_h[i].permute(0, 2, 1)) for i in range(self.hparams.num_spks)],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        est_source = torch.tanh(est_source)

        est_l = est_source.view(est_source.size(0), -1).detach()
        est_label = self.hparams.linear_1(est_l)
        est_label = self.hparams.relu(est_label)
        est_label = self.hparams.linear_2(est_label)
        # print(abc.shape)
        # exit()
        return est_source, targets, est_label

    def compute_sisdr(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    def compute_cross_entropy(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss2(predictions, targets)

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not (isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)):
            train_set = self.make_dataloader(train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs)
        if valid_set is not None and not (isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()
        self.avg_train_loss1 = 0.0
        self.avg_train_loss2 = 0.0

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss, loss1, loss2 = self.fit_batch(batch)

                    self.avg_train_loss = self.update_average(loss, self.avg_train_loss)
                    self.avg_train_loss1 = self.update_average(loss1, self.avg_train_loss1)
                    self.avg_train_loss2 = self.update_average(loss2, self.avg_train_loss2)
                    t.set_postfix(train_loss=self.avg_train_loss, loss1=self.avg_train_loss1, loss2=self.avg_train_loss2)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None and self.ckpt_interval_minutes > 0 and
                        time.time() - last_ckpt_time >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, self.avg_train_loss1, self.avg_train_loss2, epoch)
            self.avg_train_loss = 0.0
            self.avg_train_loss1 = 0.0
            self.avg_train_loss2 = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                avg_valid_loss1 = 0.0
                avg_valid_loss2 = 0.0
                with torch.no_grad():
                    for batch in tqdm(valid_set, dynamic_ncols=True, disable=not enable):
                        self.step += 1
                        loss, loss1, loss2 = self.evaluate_batch(batch, stage=sb.Stage.VALID)

                        avg_valid_loss = self.update_average(loss, avg_valid_loss)
                        avg_valid_loss1 = self.update_average(loss1, avg_valid_loss1)
                        avg_valid_loss2 = self.update_average(loss2, avg_valid_loss2)

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[sb.Stage.VALID, avg_valid_loss, avg_valid_loss1, avg_valid_loss2, epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        # mixture = batch.mix_sig
        # targets = [batch.s1_sig, batch.s2_sig]
        mixture = [batch[0], batch[1]]
        # targets = [batch[2][:,0,:],batch[2][:,1,:]]
        targets = [batch[2]]
        label = batch[3].float()

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.auto_mix_prec: # false
            with autocast():
                predictions, targets, clabel = self.compute_forward(mixture, targets, sb.Stage.TRAIN)
                loss = self.compute_sisdr(predictions, targets)
                loss2 = self.compute_cross_entropy(clabel, label)
                # print(predictions.shape)
                # print(targets.shape)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (loss < self.hparams.loss_upper_lim and loss.nelement() > 0): # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(),
                        self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:
            label = label.to(self.device)
            # print(label.shape)
            label = label.squeeze(1).float()

            predictions, targets, est_label = self.compute_forward(mixture, targets, sb.Stage.TRAIN)
            loss1 = self.compute_sisdr(predictions, targets)
            loss1 = loss1 * label
            loss2 = self.compute_cross_entropy(est_label, label)
            loss = self.hparams.alpha * loss1 + (1 - self.hparams.alpha) * loss2

            loss = loss.mean()
            loss1 = loss1.mean()
            loss2 = loss2.mean()

            if (loss < self.hparams.loss_upper_lim and loss.nelement() > 0): # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(self.modules.parameters(), self.hparams.clip_grad_norm)
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu(), loss1.detach().cpu(), loss2.detach().cpu()

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(test_set, sb.Stage.TEST, **test_loader_kwargs)
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        avg_test_loss1 = 0.0
        avg_test_loss2 = 0.0

        #---------------------------------
        if self.hparams.save_attention_weights:
            att_dir = Path(self.hparams.att_data)
            if not att_dir.exists():
                att_dir.mkdir(exist_ok=True)
            fn = 'att_weights.hdf5'
            self.inf_data_stream = h5py.File(att_dir / fn, "a")

        #---------------------------------

        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True, disable=not progressbar):
                self.step += 1
                loss, loss1, loss2 = self.evaluate_batch(batch, stage=sb.Stage.TEST, epoch=self.step)
                avg_test_loss = self.update_average(loss, avg_test_loss)
                avg_test_loss1 = self.update_average(loss1, avg_test_loss1)
                avg_test_loss2 = self.update_average(loss2, avg_test_loss2)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(self.on_stage_end, args=[sb.Stage.TEST, avg_test_loss, avg_test_loss1, avg_test_loss2, None])
        #---------------------------------
        if self.hparams.save_attention_weights:
            self.inf_data_stream.close()
        #---------------------------------
        self.step = 0

    def evaluate_batch(self, batch, stage, epoch=0):
        """Computations needed for validation/test batches"""

        mixture = [batch[0], batch[1]]

        targets = [batch[2]]

        label = batch[3].float()

        label = label.to(self.device)

        label = label.squeeze(1)

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets, est_label = self.compute_forward(mixture, targets, stage)

            loss1 = self.compute_sisdr(predictions, targets)
            loss2 = self.compute_cross_entropy(est_label, label)

            loss1 = loss1 * label
            loss = self.hparams.alpha * loss1 + (1 - self.hparams.alpha) * loss2

            loss = loss.mean()
            loss1 = loss1.mean()
            loss2 = loss2.mean()

        if stage == sb.Stage.TEST:
            if self.hparams.save_inf_data:
                # Save data
                # p = Path(self.hparams.output_folder)
                inf_dir = Path(self.hparams.inf_data)
                if not inf_dir.exists():
                    inf_dir.mkdir(exist_ok=True)
                fn = 'gw_denoise_{}.npz'.format(epoch)
                np.savez(
                    inf_dir / fn,
                    mix=batch[0].cpu().numpy(),
                    denoised=predictions.cpu().numpy(),
                    target=batch[2].cpu().numpy(),
                    est_label=est_label.cpu().numpy(),
                    label=label.cpu().numpy(),
                    #  par=batch[4].cpu().numpy()
                )

            # ______________________________________________
            if self.hparams.save_attention_weights:

                # save att weight in hdf5
                att_data = {
                    'intra_1': att_lst[0].cpu().numpy(),
                    'inter_1': att_lst[1].cpu().numpy(),
                    'intra_2': att_lst[2].cpu().numpy(),
                    'inter_2': att_lst[3].cpu().numpy(),
                }
                if epoch == 1:
                    self.inf_data_stream.create_dataset(
                        'intra_1', data=att_data['intra_1'], chunks=True, maxshape=(2, None, 162, 25, 25)
                    )
                    self.inf_data_stream.create_dataset(
                        'intra_2', data=att_data['intra_2'], chunks=True, maxshape=(2, None, 162, 25, 25)
                    )
                    self.inf_data_stream.create_dataset(
                        'inter_1', data=att_data['inter_1'], chunks=True, maxshape=(2, None, 25, 162, 162)
                    )
                    self.inf_data_stream.create_dataset(
                        'inter_2', data=att_data['inter_2'], chunks=True, maxshape=(2, None, 25, 162, 162)
                    )
                    self.inf_data_stream.create_dataset('label', data=label.cpu().numpy(), chunks=True, maxshape=(None, ))
                else:
                    for i in att_data.keys():
                        n = self.inf_data_stream[i].shape[1]
                        self.inf_data_stream[i].resize(n + att_data[i].shape[1], axis=1)
                        self.inf_data_stream[i][:, -att_data[i].shape[1]:, :, :, :] = att_data[i]

                    lab = label.cpu().numpy()
                    l = self.inf_data_stream['label'].shape[0]
                    self.inf_data_stream['label'].resize(l + lab.shape[0], axis=0)
                    self.inf_data_stream['label'][-lab.shape[0]:] = lab

        return loss.detach().cpu(), loss1.detach().cpu(), loss2.detach().cpu()

    def on_stage_end(self, stage, stage_loss, loss1, loss2, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss, 'loss1': loss1, 'loss2': loss2}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau):
                current_lr, next_lr = self.hparams.lr_scheduler([self.optimizer], epoch, stage_loss)
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": current_lr
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]},
                min_keys=["si-snr"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(test_data, **self.hparams.dataloader_opts)

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(batch.mix_sig, targets, sb.Stage.TEST)

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack([mixture] * self.hparams.num_spks, dim=-1)
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(mixture_signal, targets)
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams['cuda'])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Load dataset
    wfd = GW_SE_Dataset()
    noise = GW_SE_Dataset()
    wfd.load_waveform(DIR=hparams['data_folder'], data_fn=hparams['data_hdf5'])
    noise.load_waveform(DIR=hparams['data_folder'], data_fn=hparams['noise_hdf5'])

    wfdt_train = WaveformDatasetTorch(wfd, noise, train=True, length=hparams['training_signal_len'])
    wfdt_test = WaveformDatasetTorch(wfd, noise, train=False, length=hparams['training_signal_len'])

    train_loader = DataLoader(
        wfdt_train,
        batch_size=hparams['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32 - 1))
    )

    test_loader = DataLoader(
        wfdt_test,
        batch_size=hparams['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32 - 1))
    )

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_loader,
            test_loader,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.evaluate(test_loader, min_key="si-snr")
