"""
SimulatedDataset — manages ground-truth data on disk.

Separates the slow simulation step from the fast algorithm-evaluation step.
Ground truth (spikes, calcium, connectivity) is generated once and reused
across many preprocessing / solver / hyperparameter experiments.

Directory layout
----------------
<path>/
├── metadata.json   — all simulation parameters + creation timestamp
├── spikes.zarr/    — (N, T) binary spike trains, chunked along time
├── calcium.zarr/   — (N, T) fluorescence traces, chunked along time
└── adj_true.npy    — (N, N) synaptic weight matrix (small enough for npy)

Constructors
------------
SimulatedDataset.generate(config, path)
    Run BrunelNetwork + simulate_calcium, save to <path>, return dataset.

SimulatedDataset.load(path)
    Load an existing dataset from <path>.

SimulatedDataset.load_or_generate(config, path)
    Load if <path> exists; generate and save otherwise.
    Ideal for sweeps: first run builds the dataset, subsequent runs reuse it.

SimulatedDataset.from_hdf5(spikes_h5, adj_npy, config, save_path=None)
    Bridge for old HDF5 data from the archived scripts.
    Loads spikes + adj from disk, regenerates calcium from spikes using
    the tau/noise params in config, optionally converts everything to zarr.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import zarr

if TYPE_CHECKING:
    from ..experiments.config import ExperimentConfig


# --------------------------------------------------------------------------- #
# Zarr helpers
# --------------------------------------------------------------------------- #

def _save_zarr(arr: np.ndarray, path: Path, chunk_t: int = 10_000) -> None:
    z = zarr.open_array(
        str(path),
        mode="w",
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=(arr.shape[0], min(chunk_t, arr.shape[1])),
    )
    z[:] = arr


def _load_zarr(path: Path) -> zarr.Array:
    return zarr.open_array(str(path), mode="r")


# --------------------------------------------------------------------------- #
# SimulatedDataset
# --------------------------------------------------------------------------- #

class SimulatedDataset:
    """
    On-disk representation of one ground-truth simulation run.

    Arrays are kept as zarr (lazy, chunked) so large datasets do not
    need to fit in RAM.  Call  ds.spikes[:]  or  np.asarray(ds.spikes)
    to materialise into a numpy array.
    """

    _SPIKES_DIR  = "spikes.zarr"
    _CALCIUM_DIR = "calcium.zarr"
    _ADJ_FILE    = "adj_true.npy"
    _META_FILE   = "metadata.json"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._meta: dict | None = None

    # ------------------------------------------------------------------ #
    # Lazy properties
    # ------------------------------------------------------------------ #

    @property
    def spikes(self) -> zarr.Array:
        return _load_zarr(self.path / self._SPIKES_DIR)

    @property
    def calcium(self) -> zarr.Array:
        return _load_zarr(self.path / self._CALCIUM_DIR)

    @property
    def adj_true(self) -> np.ndarray:
        return np.load(self.path / self._ADJ_FILE)

    @property
    def metadata(self) -> dict:
        if self._meta is None:
            self._meta = json.loads((self.path / self._META_FILE).read_text())
        return self._meta

    @property
    def N(self) -> int:
        return self.spikes.shape[0]

    @property
    def T(self) -> int:
        return self.spikes.shape[1]

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def generate(
        cls,
        config: "ExperimentConfig",
        path: str | Path,
    ) -> "SimulatedDataset":
        """
        Run Brunel network + calcium simulation, save to <path>.

        Parameters
        ----------
        config : ExperimentConfig  — simulation parameters are read from here.
        path   : Directory to create (must not already exist as a dataset).
        """
        from ..simulation.brunel_network import BrunelNetwork
        from ..simulation.calcium_signal import simulate_calcium

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # -- Brunel network ------------------------------------------------
        print(f"[SimulatedDataset] Simulating Brunel network → {path.name}")
        net = BrunelNetwork(
            n_excitatory=config.n_excitatory,
            n_inhibitory=config.n_inhibitory,
            epsilon=config.epsilon,
            g=config.g,
            eta=config.eta,
            J_ex=config.J_ex,
            sim_time=config.sim_time,
            dt=config.dt,
            n_threads=config.n_threads,
            seed=config.seed,
        )
        net.build()
        net.run()
        spikes, adj_true = net.get_results()   # (N, T), (N, N)

        # -- Calcium signal ------------------------------------------------
        print(f"[SimulatedDataset] Generating calcium signal …")
        calcium = simulate_calcium(
            spikes,
            tau=config.tau,
            dt=config.dt,
            amplitude=config.amplitude,
            sigma_intra=config.sigma_intra,
            sigma_extra=config.sigma_extra,
            seed=config.seed,
        )   # (N, T)

        # -- Save ----------------------------------------------------------
        ds = cls(path)
        ds._write(spikes, calcium, adj_true, config.__dict__)
        print(f"[SimulatedDataset] Saved  N={ds.N}  T={ds.T}  → {path}")
        return ds

    @classmethod
    def load(cls, path: str | Path) -> "SimulatedDataset":
        """Load an existing dataset. Raises FileNotFoundError if missing."""
        path = Path(path)
        required = [
            path / cls._SPIKES_DIR,
            path / cls._CALCIUM_DIR,
            path / cls._ADJ_FILE,
            path / cls._META_FILE,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Dataset at '{path}' is incomplete. Missing: {missing}"
            )
        ds = cls(path)
        print(f"[SimulatedDataset] Loaded  N={ds.N}  T={ds.T}  ← {path}")
        return ds

    @classmethod
    def load_or_generate(
        cls,
        config: "ExperimentConfig",
        path: str | Path,
    ) -> "SimulatedDataset":
        """
        Load if <path> already contains a valid dataset; generate otherwise.

        This is the recommended constructor for parameter sweeps:
        the first run creates the dataset, all subsequent runs reuse it.
        """
        path = Path(path)
        meta_file = path / cls._META_FILE
        if meta_file.exists():
            return cls.load(path)
        return cls.generate(config, path)

    @classmethod
    def from_hdf5(
        cls,
        spikes_h5_path: str,
        adj_npy_path: str,
        config: "ExperimentConfig",
        save_path: str | Path | None = None,
        spikes_dataset_key: str = "spikes_trains",
    ) -> "SimulatedDataset":
        """
        Load spike data from an old HDF5 file and adj from a .npy file,
        regenerate calcium using config parameters, and optionally persist
        as a zarr dataset.

        Parameters
        ----------
        spikes_h5_path      : path to .h5 file (old BrunelNetwork.save_data())
        adj_npy_path        : path to .npy connectivity matrix
        config              : ExperimentConfig for calcium simulation params
        save_path           : if given, save the result as a zarr dataset
        spikes_dataset_key  : HDF5 dataset key (default 'spikes_trains')
        """
        from ..simulation.calcium_signal import simulate_calcium

        print(f"[SimulatedDataset] Loading spikes from HDF5: {spikes_h5_path}")
        with h5py.File(spikes_h5_path, "r") as f:
            spikes = np.asarray(f[spikes_dataset_key])   # (N, T)

        adj_true = np.load(adj_npy_path)                  # (N, N)

        print(f"[SimulatedDataset] Generating calcium from loaded spikes …")
        calcium = simulate_calcium(
            spikes,
            tau=config.tau,
            dt=config.dt,
            amplitude=config.amplitude,
            sigma_intra=config.sigma_intra,
            sigma_extra=config.sigma_extra,
            seed=config.seed,
        )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            ds = cls(save_path)
            meta = dict(config.__dict__)
            meta["source_hdf5"] = str(spikes_h5_path)
            ds._write(spikes, calcium, adj_true, meta)
            print(f"[SimulatedDataset] Converted and saved → {save_path}")
            return ds

        # In-memory dataset (no zarr — wrap into a temporary object)
        return _InMemoryDataset(spikes, calcium, adj_true)

    # ------------------------------------------------------------------ #
    # Internal write
    # ------------------------------------------------------------------ #

    def _write(
        self,
        spikes: np.ndarray,
        calcium: np.ndarray,
        adj_true: np.ndarray,
        sim_params: dict,
    ) -> None:
        _save_zarr(spikes,  self.path / self._SPIKES_DIR)
        _save_zarr(calcium, self.path / self._CALCIUM_DIR)
        np.save(self.path / self._ADJ_FILE, adj_true)

        meta = {
            "sim_params": {k: v for k, v in sim_params.items()
                           if isinstance(v, (int, float, str, bool, type(None)))},
            "shape": {"N": int(spikes.shape[0]), "T": int(spikes.shape[1])},
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.path / self._META_FILE).write_text(json.dumps(meta, indent=2))
        self._meta = meta

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        try:
            return f"SimulatedDataset(N={self.N}, T={self.T}, path='{self.path}')"
        except Exception:
            return f"SimulatedDataset(path='{self.path}')"


# --------------------------------------------------------------------------- #
# In-memory fallback (returned by from_hdf5 when no save_path given)
# --------------------------------------------------------------------------- #

class _InMemoryDataset(SimulatedDataset):
    """Wraps numpy arrays with the same interface as SimulatedDataset."""

    def __init__(
        self,
        spikes: np.ndarray,
        calcium: np.ndarray,
        adj_true: np.ndarray,
    ):
        self._spikes_arr  = spikes
        self._calcium_arr = calcium
        self._adj_arr     = adj_true

    @property
    def spikes(self):   return self._spikes_arr
    @property
    def calcium(self):  return self._calcium_arr
    @property
    def adj_true(self): return self._adj_arr
    @property
    def N(self): return self._spikes_arr.shape[0]
    @property
    def T(self): return self._spikes_arr.shape[1]
    @property
    def metadata(self): return {}
    @property
    def path(self): return Path("(in-memory)")

    def __repr__(self) -> str:
        return f"_InMemoryDataset(N={self.N}, T={self.T})"
