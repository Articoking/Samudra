#!/usr/bin/env python3

"""
Create an ensemble of Samudra rollouts with a set perturbation level.
"""

import argparse
import datetime
import logging
import os
import time
import traceback

import torch
import xarray as xr

from config import EnsembleConfig
from constants import BOUND_VARS_MAP, PROG_VARS_MAP, TensorMap, construct_metadata
from datasets import InferenceDataset
from models.samudra import Samudra
from stepper import Stepper
from perturbations import GaussianNoisePerturbator
from utils.data import Normalize, extract_wet_mask, get_inference_steps, validate_data
from utils.device import get_device, using_gpu
from utils.distributed import set_seed
from utils.logging import handle_logging, handle_warnings
from utils.directories import create_ensemble_directories

class NoisyEnsemble:
    
    # TODO: With the new EnsembleConfig class, create an InferenceDataset
    # that will be used in self.run to create the ensemble.

    def __init__(self, cfg: EnsembleConfig) -> None:
        self.device = get_device()

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.data.num_workers = 0  # Disable multi-processing on CPU
            cfg.pin_mem = False
        elif cfg.disk_mode:
            cfg.data.num_workers = torch.cuda.device_count() * cfg.data.num_workers
            cfg.pin_mem = True
        
        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting prognostic and boundary variables
        self.prognostic_vars = PROG_VARS_MAP[cfg.experiment.prognostic_vars_key]
        self.boundary_vars = BOUND_VARS_MAP[cfg.experiment.boundary_vars_key]

        self.levels = 19

        self.str_prog_vars = ", ".join([i for i in self.prognostic_vars])
        self.str_bound_vars = ", ".join([i for i in self.boundary_vars])

        logging.info(f"Prognostic variables: {self.str_prog_vars}")
        logging.info(f"Boundary variables: {self.str_bound_vars}")
        logging.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_vars)
        self.N_prog = len(self.prognostic_vars)

        self.num_in = int((cfg.data.hist + 1) * self.N_prog + self.N_bound)
        self.num_out = int((cfg.data.hist + 1) * self.N_prog)

        self.tensor_map = TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        logging.info(
            f"Number of inputs: (hist + 1) * prognostic_vars + boundary_vars "
            f"= {self.num_in}"
        )
        logging.info(
            f"Number of outputs: (hist + 1) * prognostic_vars = {self.num_out}"
        )

        # Dataloaders
        logging.info(f"Loading data")
        self.data_dir = cfg.experiment.data_dir
        self.data_path = cfg.data.data_path
        self.data_means_path = cfg.data.data_means_path
        self.data_stds_path = cfg.data.data_stds_path

        data = xr.open_zarr(os.path.join(self.data_dir, self.data_path), chunks={})
        data_mean = xr.open_dataset(
            os.path.join(self.data_dir, self.data_means_path),
            engine="zarr",
            chunks={},
        )
        data_std = xr.open_dataset(
            os.path.join(self.data_dir, self.data_stds_path),
            engine="zarr",
            chunks={},
        )
        self.data, self.data_mean, self.data_std = validate_data(
            data, data_mean, data_std
        )

        self.metadata = construct_metadata(self.data)
        self.wet, self.wet_surface = extract_wet_mask(
            self.data, self.prognostic_vars, cfg.data.hist
        )
        wet_without_hist, _ = extract_wet_mask(self.data, self.prognostic_vars, 0)

        self.normalize = Normalize.init_instance(
            self.data_mean,
            self.data_std,
            self.prognostic_vars,
            self.boundary_vars,
            wet_without_hist,
        )

        # Model
        logging.info(
            f"Instantiating model {cfg.experiment.network} from checkpoint "
            f"{cfg.ckpt_path}"
        )
        if "samudra" == cfg.experiment.network:
            if cfg.unet.ch_width[0] != self.num_in:
                logging.info(
                    f"NOTE: Changing input channels to match data "
                    f"{cfg.unet.ch_width[0]}->{self.num_in}"
                )
                cfg.unet.ch_width[0] = self.num_in
            if cfg.unet.n_out != self.num_out:
                logging.info(
                    f"NOTE: Changing output channels to match data "
                    f"{cfg.unet.n_out}->{self.num_out}"
                )
                cfg.unet.n_out = self.num_out
            model = Samudra(
                cfg.unet, hist=cfg.data.hist, wet=self.wet.to(self.device)
            ).to(self.device)
        else:
            raise NotImplementedError

        model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=torch.device("cuda"))["model"]
        )

        self.model = model

        self.network = cfg.experiment.network

        # Rollout
        self.hist = cfg.data.hist
        self.output_dir = cfg.experiment.output_dir
        self.network = cfg.experiment.network
        self.debug = cfg.debug
        self.num_workers = cfg.data.num_workers
        self.inference_time = cfg.inference
        self.time_delta = cfg.data.time_delta
        self.record_every = cfg.record_every
        self.num_model_steps_forward = cfg.num_model_steps_forward
        self.save_zarr = cfg.save_zarr
        self.model_path = cfg.ckpt_path
        self.init_inference_store()

        # Perturbation
        perturbator = GaussianNoisePerturbator(cfg)
        self.perturbator = perturbator
        

    def init_inference_store(self):
        self.num_time_steps = get_inference_steps(
            self.inference_time,
            time_delta=self.time_delta,
            hist=self.hist,
        )
        inference_data = self.data.sel(
            time=slice(self.inference_time.start_time, self.inference_time.end_time)
        )
        self.inference_dataset = InferenceDataset(
            inference_data,
            self.prognostic_vars,
            self.boundary_vars,
            self.wet,
            self.wet_surface,
            self.hist,
        )

    def run(self) -> None:
        # TODO: Use Rollout class and the InferenceDataset from __init__ to
        # do a `standalone_inference` for each member of the ensemble, then
        # save to zarr in some format
        start_time = time.time()
        self.ensemble_inference()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Rollout time {total_time_str}")

    @torch.no_grad()
    def ensemble_inference(self):
        """
        Do inference for all members of the ensemble.
        """
        self.model.eval()
        
        logging.info(f"Doing inference for {self.perturbator.n_members} "\
                     "ensemble members")
        logging.info(f"num_model_steps_forward: {self.num_model_steps_forward}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--subname", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--n_members", type=int, required=False)
    parser.add_argument("--save_zarr", default=False, action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.subname:
        overrides["sub_name"] = args.subname
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    if args.save_zarr:
        overrides["save_zarr"] = args.save_zarr
    if args.n_members:
        overrides["n_members"] = args.n_members

    # Load config from YAML
    cfg = EnsembleConfig.from_yaml(args.config, overrides)

    if cfg.experiment.output_dir.exists():
        raise ValueError(
            f"Output directory {cfg.experiment.output_dir} already exists, "
            "please delete it or use a different expt directory"
        )
    cfg.experiment.output_dir.mkdir()
    create_ensemble_directories(cfg)

    handle_logging(cfg)
    handle_warnings()

    Evaluator = NoisyEnsemble(cfg)
    try:
        Evaluator.run()
    except Exception as e:
        # log traceback
        logging.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()