"""
Brunel balanced random network simulator using NEST.

Implements the classic sparse random network of integrate-and-fire neurons
(Brunel 2000). Outputs spike trains and ground-truth connectivity matrix.

Usage
-----
    from calcium_ar.simulation import BrunelNetwork

    net = BrunelNetwork(
        n_excitatory=800,
        n_inhibitory=200,
        epsilon=0.1,
        g=5.0,
        eta=2.0,
        sim_time=1000.0,   # ms
        dt=0.1,            # ms
        n_threads=4,
    )
    net.build()
    net.run()
    spikes, adj = net.get_results()   # (N, T) binary, (N, N) float
    net.save("spikes.h5", "connectivity.npy")
"""

import time
import numpy as np
import h5py


class BrunelNetwork:
    """
    Balanced random spiking network (Brunel 2000) driven by NEST.

    Parameters
    ----------
    n_excitatory : int
        Number of excitatory neurons.
    n_inhibitory : int
        Number of inhibitory neurons.
    epsilon : float
        Connection probability (fraction of incoming connections per neuron).
    g : float
        Ratio of inhibitory to excitatory synaptic weight (|J_in| = g * J_ex).
    eta : float
        External drive relative to threshold rate.
    J_ex : float
        Excitatory synaptic weight (mV).
        Scale rule: J_ex × C_E should equal ~80 (Brunel 2000 reference: N=10,000,
        C_E=800, J=0.1 → J×C_E=80). For n_excitatory=800 (C_E=80): J_ex=1.0.
        For n_excitatory=1000 (C_E=100): J_ex=0.8. Etc.
    delay : float
        Synaptic transmission delay (ms).
    tau_m : float
        Membrane time constant (ms).
    V_th : float
        Spike threshold (mV).
    sim_time : float
        Total simulation duration (ms).
    dt : float
        Simulation time resolution (ms).
    n_threads : int
        Number of CPU threads for NEST.
    seed : int
        Seed for NEST's random number generator. Together with n_threads this
        fully determines the network wiring and spike trains (NEST's RNG stream
        depends on the number of threads, so reproducibility needs both fixed).
    """

    def __init__(
        self,
        n_excitatory: int = 800,
        n_inhibitory: int = 200,
        epsilon: float = 0.1,
        g: float = 5.0,
        eta: float = 2.0,
        J_ex: float = 1.0,   # 0.1 × (800_Brunel / C_E_here=80) = 1.0 mV
        delay: float = 1.5,
        tau_m: float = 20.0,
        V_th: float = 20.0,
        V_reset: float = 0.0,   # Brunel 2000 uses 10.0; 0.0 kept as the default
                                # so every existing result stays reproducible.
        sim_time: float = 1000.0,
        dt: float = 0.1,
        n_threads: int = 1,
        seed: int = 42,
    ):
        self.NE = n_excitatory
        self.NI = n_inhibitory
        self.N = n_excitatory + n_inhibitory
        self.epsilon = epsilon
        self.g = g
        self.eta = eta
        self.J_ex = J_ex
        self.J_in = -g * J_ex
        self.delay = delay
        self.tau_m = tau_m
        self.V_th = V_th
        self.V_reset = V_reset
        self.sim_time = sim_time
        self.dt = dt
        self.n_threads = n_threads
        self.seed = seed
        if seed < 1:
            raise ValueError(
                f"seed must be >= 1 (NEST's RNG seed range is 1..2^32-1); got {seed}"
            )

        # Integer in-degrees
        self.CE = max(1, int(epsilon * n_excitatory))
        self.CI = max(1, int(epsilon * n_inhibitory))

        self._spikes: np.ndarray | None = None
        self._adj: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poisson_rate(self) -> float:
        nu_th = self.V_th / (self.J_ex * self.CE * self.tau_m)
        nu_ex = self.eta * nu_th
        return 1000.0 * nu_ex * self.CE  # Hz → spikes/s, then *CE

    def _neuron_params(self) -> dict:
        return {
            "tau_m": self.tau_m,
            "t_ref": 2.0,
            "V_m": 0.0,
            "V_th": self.V_th,
            "V_reset": self.V_reset,
            "E_L": 0.0,
            "C_m": 1.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Initialise NEST, create populations and connect them."""
        import nest

        nest.ResetKernel()
        nest.SetKernelStatus({
            "local_num_threads": self.n_threads,
            "resolution": self.dt,
            "rng_seed": self.seed,   # makes the network wiring + spike trains reproducible
            "print_time": False,
            "overwrite_files": True,
        })

        neuron_params = self._neuron_params()
        nest.SetDefaults("iaf_psc_delta", neuron_params)

        self._nodes_ex = nest.Create("iaf_psc_delta", self.NE, params=neuron_params)
        self._nodes_in = nest.Create("iaf_psc_delta", self.NI, params=neuron_params)
        self._all_nodes = self._nodes_ex + self._nodes_in

        noise = nest.Create("poisson_generator", params={"rate": self._poisson_rate()})

        nest.CopyModel("static_synapse", "excitatory", {"weight": self.J_ex, "delay": self.delay})
        nest.CopyModel("static_synapse", "inhibitory", {"weight": self.J_in, "delay": self.delay})

        # External drive
        nest.Connect(noise, self._nodes_ex, syn_spec="excitatory")
        nest.Connect(noise, self._nodes_in, syn_spec="excitatory")

        # Recurrent connections
        nest.Connect(
            self._nodes_ex, self._all_nodes,
            conn_spec={"rule": "fixed_indegree", "indegree": self.CE},
            syn_spec="excitatory",
        )
        nest.Connect(
            self._nodes_in, self._all_nodes,
            conn_spec={"rule": "fixed_indegree", "indegree": self.CI},
            syn_spec="inhibitory",
        )

        # Spike recorder
        self._recorder = nest.Create("spike_recorder")
        nest.Connect(self._all_nodes, self._recorder)

    def run(self, densify: bool = True) -> None:
        """Run the simulation.

        densify=True (default) builds the dense (N, T) spike matrix. For very
        long recordings set densify=False and read spikes sparsely via
        get_spike_events() — the dense matrix would not fit in RAM."""
        import nest

        t0 = time.time()
        nest.Simulate(self.sim_time)
        elapsed = time.time() - t0
        print(f"Simulation finished in {elapsed:.1f}s")
        if densify:
            self._extract_results()

    def get_spike_events(self) -> tuple[np.ndarray, np.ndarray]:
        """Return spikes as sparse events (no dense (N, T) array).

        Returns (neuron_index, spike_time_ms): neuron_index is 0-based in
        [0, N), spike_time_ms is the event time in ms."""
        import nest
        ev = nest.GetStatus(self._recorder, "events")[0]
        senders = np.asarray(ev["senders"], dtype=np.int64)
        times = np.asarray(ev["times"], dtype=float)
        base = int(self._all_nodes.tolist()[0])          # first neuron GID
        idx = senders - base
        return idx, times

    def get_adjacency(self) -> np.ndarray:
        """Ground-truth (N, N) connectivity, without densifying spikes.
        adj[s, t] = synaptic weight from source s to target t (0-indexed)."""
        import nest
        conn = nest.GetConnections(source=self._all_nodes, target=self._all_nodes)
        base = int(self._all_nodes.tolist()[0])
        sources = np.array(nest.GetStatus(conn, "source")) - base
        targets = np.array(nest.GetStatus(conn, "target")) - base
        weights = np.array(nest.GetStatus(conn, "weight"))
        adj = np.zeros((self.N, self.N))
        adj[sources, targets] = weights
        return adj

    def _extract_results(self) -> None:
        import nest

        events = nest.GetStatus(self._recorder, "events")[0]
        sp_times = events["times"]       # ms, float
        sp_senders = events["senders"]   # neuron GID, 1-indexed

        T = int(self.sim_time / self.dt)
        time_bins = np.arange(0, self.sim_time + self.dt, self.dt)
        neuron_bins = np.arange(1, self.N + 2)  # 1-indexed GIDs

        self._spikes, _, _ = np.histogram2d(
            sp_senders, sp_times,
            bins=[neuron_bins, time_bins],
        )  # shape (N, T)

        # Ground-truth connectivity matrix
        conn_tuple = nest.GetConnections(
            source=self._all_nodes, target=self._all_nodes
        )
        sources = np.array(nest.GetStatus(conn_tuple, "source")) - 1  # 0-indexed
        targets = np.array(nest.GetStatus(conn_tuple, "target")) - 1
        weights = np.array(nest.GetStatus(conn_tuple, "weight"))

        self._adj = np.zeros((self.N, self.N))
        self._adj[sources, targets] = weights

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (spikes, adjacency_matrix) after run().

        Returns
        -------
        spikes : np.ndarray, shape (N, T)
            Binary spike train matrix (1 where a spike occurred).
        adj : np.ndarray, shape (N, N)
            Synaptic weight matrix (ground truth connectivity).
        """
        if self._spikes is None:
            raise RuntimeError("Call build() and run() first.")
        return self._spikes, self._adj

    def save(self, spikes_path: str, adj_path: str) -> None:
        """Persist spikes (HDF5) and adjacency matrix (npy)."""
        spikes, adj = self.get_results()
        with h5py.File(spikes_path, "w") as f:
            f.create_dataset("spikes", data=spikes)
        np.save(adj_path, adj)
        print(f"Saved spikes → {spikes_path}")
        print(f"Saved connectivity → {adj_path}")
