#!/usr/bin/env python3

"""
Creates an ensemble of Samudra rollouts with a set perturbation level.
"""

import argparse

from config import EnsembleConfig
from utils.logging import handle_logging, handle_warnings
from utils.directories import create_ensemble_directories

class NoisyEnsemble:

    def __init__(self):
        # TODO: Create new config file that extends EvalConfig in `config.py`
        # This class should include reading a sigma for the noise, or whatever
        # parameter we need to use to create the ensembles.
        
        # TODO: With the new EnsembleConfig class, create an InferenceDataset
        # that will be used in self.run to create the ensemble.
        pass

    def run(self) -> None:
        # TODO: Use Rollout class and the InferenceDataset from __init__ to
        # do a `standalone_inference` for each member of the ensemble, then
        # save to zarr in some format
        pass

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

if __name__ == "__main__":
    main()