"""
ParameterSweep — run the full pipeline over a grid of parameter values
and collect results into a summary DataFrame.

Supports:
  - Single-parameter sweep:  {'lag': [1, 5, 10, 20, 50]}
  - Multi-parameter grid:    {'lag': [5, 10], 'lam': [0.1, 1.0, 10.0]}
    → Cartesian product (all combinations)
  - Parallel execution via joblib (n_jobs > 1) — works on laptop and HPC

Results are saved inside  {sweep_dir}/  so every individual run can be
reloaded with persistence.load_result().

Usage
-----
    from calcium_ar.experiments import ExperimentConfig, ParameterSweep

    base = ExperimentConfig(solver='ridge', sim_time=5000.0)
    sweep = ParameterSweep(
        params={'lag_ms': [1.0, 5.0, 10.0, 20.0], 'lam': [0.1, 1.0, 10.0]},
        base_config=base,
        sweep_dir='results/sweeps/lag_vs_lam',
    )
    df = sweep.run(n_jobs=4)   # 4 parallel workers
    print(df.sort_values('correlation', ascending=False))
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .runner import run_single
from .persistence import load_sweep_results


class ParameterSweep:
    """
    Grid search over one or more ExperimentConfig parameters.

    Parameters
    ----------
    params : dict[str, list]
        Parameter names → list of values to try.
        All combinations (Cartesian product) will be run.
    base_config : ExperimentConfig
        Base configuration; swept parameters override its fields.
    sweep_dir : str
        Parent directory that will contain one sub-directory per run.
    """

    def __init__(
        self,
        params: dict[str, list[Any]],
        base_config: ExperimentConfig,
        sweep_dir: str = "results/sweep",
    ):
        for key in params:
            if not hasattr(base_config, key):
                raise ValueError(
                    f"'{key}' is not a field of ExperimentConfig. "
                    f"Valid fields: {[f.name for f in __import__('dataclasses').fields(base_config)]}"
                )
        self.params     = params
        self.base       = base_config
        self.sweep_dir  = Path(sweep_dir)

    # ------------------------------------------------------------------ #
    # Config grid
    # ------------------------------------------------------------------ #

    def _make_configs(self) -> list[ExperimentConfig]:
        """Cartesian product of all parameter values."""
        keys   = list(self.params.keys())
        values = list(self.params.values())
        configs = []
        for combo in itertools.product(*values):
            overrides = dict(zip(keys, combo))
            # Build a descriptive run name
            tag = "_".join(f"{k}={v}" for k, v in overrides.items())
            name = f"{self.base.name}__{tag}"
            cfg = self.base.replace(
                **overrides,
                name=name,
                output_dir=str(self.sweep_dir),
            )
            configs.append(cfg)
        return configs

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self, n_jobs: int = 1):
        """
        Execute all configurations and return a summary DataFrame.

        Parameters
        ----------
        n_jobs : int
            Number of parallel workers.
            1  → sequential (safe on any system, easier to debug)
            -1 → use all available CPU cores
            N  → use N cores (good for HPC nodes)

        Returns
        -------
        pd.DataFrame  — one row per run, columns = config fields + metrics.
        """
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        configs = self._make_configs()
        print(
            f"Starting sweep: {len(configs)} configurations "
            f"across {list(self.params.keys())}"
        )

        if n_jobs == 1:
            for i, cfg in enumerate(configs, 1):
                print(f"\n--- Run {i}/{len(configs)}: {cfg.name} ---")
                run_single(cfg)
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                raise ImportError(
                    "joblib is required for parallel sweeps: pip install joblib"
                )
            Parallel(n_jobs=n_jobs)(
                delayed(run_single)(cfg) for cfg in configs
            )

        return load_sweep_results(self.sweep_dir)

    # ------------------------------------------------------------------ #
    # Convenience: reload without re-running
    # ------------------------------------------------------------------ #

    def load_results(self):
        """Load previously saved sweep results without re-running."""
        return load_sweep_results(self.sweep_dir)

    def __repr__(self) -> str:
        n = len(list(itertools.product(*self.params.values())))
        return (
            f"ParameterSweep(params={self.params}, "
            f"n_configs={n}, sweep_dir='{self.sweep_dir}')"
        )
