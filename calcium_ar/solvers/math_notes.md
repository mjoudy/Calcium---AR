# Neural Connectivity Inference — Mathematical Notes

## Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $N$ | scalar | number of neurons |
| $T$ | scalar | number of time steps |
| $\ell$ | scalar | AR lag parameter |
| $\mathbf{X}_t$ | $N \times (T-\ell)$ | preprocessed signals at time $t$ |
| $\mathbf{X}_{t-\ell}$ | $N \times (T-\ell)$ | same signals shifted by lag $\ell$ |
| $\mathbf{A}$ | $N \times N$ | connectivity matrix (what we want) |
| $\mathbf{R}$ | $N \times (T-\ell)$ | residual matrix $\mathbf{X}_t - \mathbf{A}\mathbf{X}_{t-\ell}$ |
| $\|\cdot\|_F$ | | Frobenius norm $\sqrt{\sum_{ij} a_{ij}^2}$ |
| $\|\cdot\|_1$ | | entrywise L1 norm $\sum_{ij} \lvert a_{ij} \rvert$ |
| $\mathbf{g}$ | $N \times N$ | gradient of the loss w.r.t. $\mathbf{A}$ |

---

## 1. The model

Each neuron's state at time $t$ is assumed to be a linear combination of all
neurons' states at time $t - \ell$:

$$\mathbf{X}_t = \mathbf{A} \, \mathbf{X}_{t-\ell} + \boldsymbol{\varepsilon}$$

where $\boldsymbol{\varepsilon}$ is noise. The goal is to estimate $\mathbf{A}$
from observations $\mathbf{X}$. Each column of $\mathbf{X}_t$ is one time step;
each row is one neuron.

---

## 2. Ordinary Least Squares (OLS) — `chunked_ols.py`

**Objective** — minimise the squared reconstruction error:

$$\min_{\mathbf{A}} \; \|\mathbf{X}_t - \mathbf{A}\,\mathbf{X}_{t-\ell}\|_F^2$$

**Derivation** — take the matrix gradient and set to zero:

$$\frac{\partial}{\partial \mathbf{A}} \|\mathbf{X}_t - \mathbf{A}\,\mathbf{X}_{t-\ell}\|_F^2
= -2\,\underbrace{(\mathbf{X}_t - \mathbf{A}\,\mathbf{X}_{t-\ell})}_{\mathbf{R}}\,\mathbf{X}_{t-\ell}^T = \mathbf{0}$$

Rearranging gives the **normal equations**:

$$\mathbf{A} \underbrace{\mathbf{X}_{t-\ell}\mathbf{X}_{t-\ell}^T}_{\mathbf{C}_{xx}}
= \underbrace{\mathbf{X}_t \mathbf{X}_{t-\ell}^T}_{\mathbf{C}_{yx}}$$

$$\boxed{\mathbf{A} = \mathbf{C}_{yx}\,\mathbf{C}_{xx}^{-1}}$$

Both $\mathbf{C}_{xx}$ and $\mathbf{C}_{yx}$ are $N \times N$ — small regardless of $T$.

**Connection to Moore–Penrose pseudoinverse** — via the SVD
$\mathbf{X}_{t-\ell} = \mathbf{U}\mathbf{S}\mathbf{V}^T$:

$$\mathbf{C}_{xx}^{-1}\mathbf{X}_{t-\ell}^T = (\mathbf{V}\mathbf{S}^2\mathbf{V}^T)^{-1}\mathbf{V}\mathbf{S}\mathbf{U}^T
= \mathbf{V}\mathbf{S}^{-1}\mathbf{U}^T = \mathbf{X}_{t-\ell}^+$$

The normal equations give the pseudoinverse implicitly — without ever
constructing the $T \times N$ matrix $\mathbf{U}$.

---

## 3. Why chunking is exact (not an approximation)

$\mathbf{C}_{xx}$ is a sum over columns (time steps):

$$\mathbf{C}_{xx} = \sum_{t=\ell}^{T} \mathbf{x}_{t-\ell}\,\mathbf{x}_{t-\ell}^T$$

Sums decompose over any partition of the time axis into $K$ chunks:

$$\mathbf{C}_{xx} = \sum_{k=1}^{K} \mathbf{X}_{t-\ell}^{(k)}\,\bigl(\mathbf{X}_{t-\ell}^{(k)}\bigr)^T$$

Each chunk $\mathbf{X}_{t-\ell}^{(k)}$ is $N \times \text{chunk\_size}$ and its
outer product is $N \times N$. Accumulated over $K$ chunks, the result is
**numerically identical** to loading all $T$ columns at once. No approximation.

---

## 4. Ridge Regression (L2) — `chunked_ridge.py`

**Objective:**

$$\min_{\mathbf{A}} \; \|\mathbf{X}_t - \mathbf{A}\,\mathbf{X}_{t-\ell}\|_F^2
+ \lambda\,\|\mathbf{A}\|_F^2$$

**Solution** — the extra $\lambda\|\mathbf{A}\|_F^2$ term adds $2\lambda\mathbf{A}$
to the gradient, shifting the normal equations to:

$$\boxed{\mathbf{A} = \mathbf{C}_{yx}\,(\mathbf{C}_{xx} + \lambda\mathbf{I})^{-1}}$$

Adding $\lambda\mathbf{I}$ shifts all eigenvalues of $\mathbf{C}_{xx}$ up by $\lambda$.
In terms of singular values this gives the **Tikhonov-regularised pseudoinverse**:

$$\mathbf{V}\,\mathrm{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right)\mathbf{U}^T$$

Near-zero $\sigma_i$ (caused by correlated neurons) no longer blow up.
All entries of $\mathbf{A}$ are shrunk toward zero but **none reach exactly zero**
— the matrix remains dense.

Setting $\lambda = 0$ recovers plain OLS. The chunk accumulation is identical;
only the final solve changes.

---

## 5. PyTorch Method 1 — GPU Normal Equations — `torch_normal_eq.py`

**Mathematical content:** identical to chunked Ridge (Section 4).

$$\mathbf{A} = \mathbf{C}_{yx}\,(\mathbf{C}_{xx} + \lambda\mathbf{I})^{-1}$$

**What changes vs numpy:** the computation device.

Each chunk multiply `x_prev @ x_prev.T` is dispatched to **cuBLAS** on the GPU
instead of BLAS on the CPU. A modern GPU has thousands of CUDA cores that
execute floating-point multiplications in parallel (SIMD), whereas a CPU has
8–64 cores. The asymptotic complexity $O(N^2 \cdot \text{chunk\_size})$ is the same;
the constant is 10–100× smaller on GPU.

The final solve `torch.linalg.solve` dispatches to **cuSOLVER** on GPU or
**LAPACK** on CPU — both exact, both on an $(N \times N)$ matrix.

**Memory model:**

$$\underbrace{\text{chunk on GPU VRAM}}_{\approx 100\,\text{MB}}
+ \underbrace{\mathbf{C}_{xx},\,\mathbf{C}_{yx}}_{\approx 25\,\text{MB}}
+ \underbrace{\mathbf{A}}_{\approx 12.5\,\text{MB}}$$

The full $T \times N$ data never appears anywhere. Each chunk is transferred from
CPU RAM → GPU VRAM via PCIe, processed, and freed.

**When to prefer over numpy:** any time an HPC node has a GPU. On CPU,
`torch_normal_eq` and `chunked_ridge` take the same time.

---

## 6. PyTorch Method 2 — Mini-batch Gradient Descent — `torch_minibatch.py`

### 6.1 Objective (Lasso / L1)

$$\mathcal{L}(\mathbf{A}) = \|\mathbf{X}_t - \mathbf{A}\,\mathbf{X}_{t-\ell}\|_F^2
+ \lambda\,\|\mathbf{A}\|_1$$

The L1 term has **no closed-form solution** because $\lvert a \rvert$ is not
differentiable at $a = 0$. We instead minimise iteratively.

### 6.2 Full gradient

Taking the matrix derivative of each term:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}}
= \underbrace{-2\,\mathbf{R}\,\mathbf{X}_{t-\ell}^T}_{\text{reconstruction term}}
+ \underbrace{\lambda\,\mathrm{sign}(\mathbf{A})}_{\text{L1 subgradient}}$$

where $\mathbf{R} = \mathbf{X}_t - \mathbf{A}\mathbf{X}_{t-\ell}$ and
$\mathrm{sign}(\mathbf{A})$ is applied element-wise ($+1$, $-1$, or $0$ at zero).
The subgradient replaces the gradient where the derivative does not exist.

### 6.3 Mini-batch stochastic approximation

Computing the full gradient requires summing $\mathbf{R}^{(k)}\bigl(\mathbf{X}_{t-\ell}^{(k)}\bigr)^T$
over all $K$ chunks — equivalent to one OLS pass. Instead, we use a single chunk
per update and treat it as a noisy estimate of the full gradient:

$$\mathbf{g}^{(k)} = -2\,\mathbf{R}^{(k)}\,\bigl(\mathbf{X}_{t-\ell}^{(k)}\bigr)^T
+ \lambda\,\mathrm{sign}(\mathbf{A})$$

This is an **unbiased estimator** of the full gradient:

$$\mathbb{E}_k\bigl[\mathbf{g}^{(k)}\bigr] = \frac{\partial \mathcal{L}}{\partial \mathbf{A}}$$

because each chunk is drawn from the same population of time steps. This is the
theoretical justification for mini-batch gradient descent.

### 6.4 Adam optimizer

Plain gradient descent uses a fixed step size $\eta$. **Adam** (Adaptive Moment
Estimation) maintains a separate adaptive learning rate for every element of
$\mathbf{A}$ by tracking the running mean and variance of past gradients.

Let $g_\tau$ denote the gradient at step $\tau$. Adam maintains:

$$\mathbf{m}_\tau = \beta_1\,\mathbf{m}_{\tau-1} + (1 - \beta_1)\,\mathbf{g}_\tau
\quad\text{(first moment — gradient mean)}$$

$$\mathbf{v}_\tau = \beta_2\,\mathbf{v}_{\tau-1} + (1 - \beta_2)\,\mathbf{g}_\tau^2
\quad\text{(second moment — gradient variance)}$$

Bias-corrected estimates (compensate for zero initialisation):

$$\hat{\mathbf{m}}_\tau = \frac{\mathbf{m}_\tau}{1 - \beta_1^\tau}
\qquad
\hat{\mathbf{v}}_\tau = \frac{\mathbf{v}_\tau}{1 - \beta_2^\tau}$$

Parameter update:

$$\mathbf{A} \;\leftarrow\; \mathbf{A} - \eta\,\frac{\hat{\mathbf{m}}_\tau}{\sqrt{\hat{\mathbf{v}}_\tau} + \varepsilon}$$

Default values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

**Why Adam over plain SGD:** entries of $\mathbf{A}$ that receive consistently
large gradients get a smaller effective step (preventing oscillation); entries
with small or rare gradients get a larger effective step (faster convergence).
This is critical for sparse problems where most entries of $\mathbf{A}$ are zero
and only a small fraction receive large gradients.

### 6.5 Multiple epochs and convergence

One **epoch** = one full pass over all $K$ chunks = $K$ gradient steps.
After one epoch, the gradient estimate has been computed from every time step
once — analogous to one OLS pass.

The loss after epoch $e$ is a decreasing function of $e$ under standard
conditions (Lipschitz-smooth loss, bounded gradients). Convergence is monitored
by watching the epoch loss plateau:

$$\mathcal{L}^{(e)} \to \mathcal{L}^*
\quad \text{as} \quad e \to \infty$$

In practice 5–20 epochs is sufficient. PyTorch's autograd computes
$\partial \mathcal{L}/\partial \mathbf{A}$ automatically by reverse-mode
differentiation through the forward computation graph — you do not compute
the gradient manually.

### 6.6 L1 geometry and sparsity

The L1 ball $\{\mathbf{A} : \|\mathbf{A}\|_1 \leq c\}$ is a high-dimensional
cross-polytope whose extreme points lie on the coordinate axes. When the
unconstrained OLS minimum lies outside the ball, the constrained solution
lands on a corner or edge — forcing entire rows, columns, or individual
entries to **exactly zero**. This is why L1 produces sparse $\mathbf{A}$
while L2 only shrinks.

---

## 7. PyTorch Method 3 — nn.Linear — `torch_linear_layer.py`

### 7.1 Mathematical equivalence to Method 2

The objective, gradient, and Adam update are **identical** to Section 6.
The only difference is how $\mathbf{A}$ is represented in code.

### 7.2 nn.Linear convention and orientation

`nn.Linear(N, N, bias=False)` stores weight $\mathbf{W}$ of shape $(N, N)$ and
defines the forward pass as:

$$\text{output} = \text{input} \cdot \mathbf{W}^T$$

In our code, `input` is $\mathbf{X}_{t-\ell}^T$ — the data transposed so that
**time steps are rows** (shape: $\text{chunk} \times N$). Then:

$$\text{output} = \mathbf{X}_{t-\ell}^T \cdot \mathbf{W}^T
= \bigl(\mathbf{W}\,\mathbf{X}_{t-\ell}\bigr)^T
= \mathbf{X}_t^T$$

So $\mathbf{W} = \mathbf{A}$ exactly, and `model.linear.weight` is your
connectivity matrix directly. The `.T` in the input is only on the
$N \times \text{chunk\_size}$ chunk — a small and cheap transposition.

### 7.3 Why nn.Module instead of a raw tensor

| Aspect | Raw tensor (`torch_minibatch`) | `nn.Linear` (`torch_linear_layer`) |
|--------|-------------------------------|-------------------------------------|
| Parameter tracking | Manual `requires_grad=True` | Automatic via `nn.Module` |
| Optimizer registration | `optim.Adam([A])` | `optim.Adam(model.parameters())` |
| Save / load weights | `torch.save(A)` | `torch.save(model.state_dict())` |
| Add second lag | Declare `A2`, pass to optimizer | Add `self.linear2 = nn.Linear(N,N)` |
| Add nonlinearity | Manual | `nn.ReLU()`, `nn.Tanh()`, etc. |
| Constrain weights | Manual clamp after each step | Override `forward()` |

### 7.4 Extensions that become natural with nn.Module

**Multiple lags** — model $\mathbf{X}_t = \mathbf{A}_1\mathbf{X}_{t-1} + \mathbf{A}_2\mathbf{X}_{t-2}$:

```python
self.lag1 = nn.Linear(N, N, bias=False)
self.lag2 = nn.Linear(N, N, bias=False)

def forward(self, x_prev1, x_prev2):
    return self.lag1(x_prev1) + self.lag2(x_prev2)
```

**Non-negativity** (excitatory-only connections):

```python
def forward(self, x):
    return nn.functional.linear(x, self.linear.weight.clamp(min=0))
```

**Symmetric connectivity** (undirected network assumption):

```python
def forward(self, x):
    W = (self.linear.weight + self.linear.weight.T) / 2
    return nn.functional.linear(x, W)
```

These extensions are impossible or cumbersome with a raw parameter tensor.

---

## 8. Comparison of all implemented methods

| | OLS | Ridge | PyTorch Normal Eq. | PyTorch Mini-batch | PyTorch nn.Linear |
|---|:---:|:-----:|:------------------:|:------------------:|:-----------------:|
| File | `chunked_ols` | `chunked_ridge` | `torch_normal_eq` | `torch_minibatch` | `torch_linear_layer` |
| Penalty | None | L2 | L2 optional | L1 | L1 |
| Sparse $\mathbf{A}$ | No | No | No | **Yes** | **Yes** |
| Exact solution | Yes | Yes | Yes | No (iterative) | No (iterative) |
| Data passes | 1 | 1 | 1 | $n_\text{epochs}$ | $n_\text{epochs}$ |
| GPU benefit | No | No | **Large** | Moderate | Moderate |
| Laptop feasible | Yes | Yes | Yes | Yes | Yes |
| Extensible architecture | No | No | No | No | **Yes** |
| PyTorch autograd | No | No | No | Yes | Yes |

---

## 9. Summary of the full loss landscape

All five methods minimise variants of the same underlying objective:

$$\mathcal{L}(\mathbf{A}) = \underbrace{\|\mathbf{X}_t - \mathbf{A}\,\mathbf{X}_{t-\ell}\|_F^2}_{\text{reconstruction}}
+ \underbrace{\lambda_2\,\|\mathbf{A}\|_F^2}_{\text{Ridge term}}
+ \underbrace{\lambda_1\,\|\mathbf{A}\|_1}_{\text{Lasso term}}$$

| Method | $\lambda_1$ | $\lambda_2$ | Algorithm |
|--------|:-----------:|:-----------:|-----------|
| OLS | 0 | 0 | Normal equations (one pass) |
| Ridge | 0 | $> 0$ | Regularised normal equations (one pass) |
| PyTorch Normal Eq. | 0 | $\geq 0$ | Same, on GPU |
| PyTorch Mini-batch | $> 0$ | 0 | Stochastic gradient descent + Adam |
| PyTorch nn.Linear | $> 0$ | 0 | Same, inside Module wrapper |

ElasticNet (not yet implemented) activates both $\lambda_1 > 0$ and
$\lambda_2 > 0$ simultaneously, combining sparsity with stability.

---

*Convert to PDF:*
```bash
pandoc math_notes.md -o math_notes.pdf --pdf-engine=pdflatex
```
