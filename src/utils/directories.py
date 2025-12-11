from pathlib import Path

def create_ensemble_directories(cfg):
    """
    Creates a directory for each member of an ensemble.

    All directories are created in `cfg.experiment.output_dir`.
    Directories are named numerically from 0 to `cfg.n_members-1`.
    Each directory is zero-padded so all of them have equal-length names.
    
    Parameters
    ----------
    cfg : EnsembleConfig
        Configuration file.
    """
    output_dir: Path = cfg.experiment.output_dir
    n_members = cfg.n_members
    n_digits = len(str(n_members-1))
    for idx in range(0, n_members):
        member_path = output_dir / str(idx).zfill(n_digits)
        member_path.mkdir()