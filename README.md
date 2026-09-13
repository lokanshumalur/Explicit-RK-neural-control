# Beyond Euler Method: High-Order Gradient Fidelity for Continuous-Time Deep Control

**Nadav Azran** · **Lokanshu Malur** · **Yaacov Kopeliovich** · **Michael Pokojovy**

This repository is the official implementation of the neural optimal control framework reported in the paper. The framework embeds higher-order Runge-Kutta (RK) discretizations directly into the neural network training loop, parameterizing the control policy as a state-feedback MLP and leveraging automatic differentiation through the numerical integrator for end-to-end gradient computation.

The approach is validated on three benchmark problems of increasing dimensionality and dynamical complexity:

1. PK/PD tumor growth control
2. Quadrotor stabilization
3. Lorenz-96 chaotic system stabilization

## Repository Structure

```text
├── README.md
├── LICENSE.md
│
├── PKPDNN.py
├── quadrotorNN.py
├── lorenzNN.py
│
├── analytical_pmp.py
├── quadrotor_casadi.py
├── lorenz_casadi.py
│
├── convergence_orderPKPD.py
├── convergence_orderQUADROTOR.py
├── convergence_orderLORENZ.py
│
└── run_sweep.sh
```

### Main Neural Control Scripts

- `PKPDNN.py` — PK/PD neural controller using Euler, Midpoint, and RK4
- `quadrotorNN.py` — Quadrotor neural controller using Euler, Midpoint, and RK4
- `lorenzNN.py` — Lorenz-96 neural controller using Euler, Midpoint, and RK4

### Reference-Solution Scripts

- `analytical_pmp.py` — Pontryagin's Maximum Principle benchmark for PK/PD
- `quadrotor_casadi.py` — CasADi/IPOPT reference solution for the quadrotor problem
- `lorenz_casadi.py` — CasADi/IPOPT reference solution for the Lorenz-96 problem

### Convergence Analysis

- `convergence_orderPKPD.py` — Convergence analysis for PK/PD
- `convergence_orderQUADROTOR.py` — Convergence analysis for Quadrotor
- `convergence_orderLORENZ.py` — Convergence analysis for Lorenz-96

### Experiment Automation

- `run_sweep.sh` — Sequential multi-seed experiment runner for the three benchmark problems

## Dependencies

The primary dependencies are:

```text
tensorflow
numpy
scipy
matplotlib
casadi
```

Install them with:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install tensorflow numpy scipy matplotlib casadi
```

### Hardware Acceleration

#### Apple Silicon

For supported Apple Silicon systems, TensorFlow Metal can be installed with:

```bash
pip install tensorflow-metal
```

#### NVIDIA GPUs

TensorFlow can use supported CUDA-enabled NVIDIA GPUs when the appropriate CUDA environment is installed.

#### CPU

All scripts can also run on CPU, although the larger experiments may require substantially more time.

The neural-network scripts use Float64 precision and automatically configure available TensorFlow GPU devices.

## Running the Experiments

### Benchmark 1: PK/PD Tumor Growth

Train the neural controllers:

```bash
python PKPDNN.py
```

Generate the Pontryagin's Maximum Principle reference solution:

```bash
python analytical_pmp.py
```

Run the convergence analysis:

```bash
python convergence_orderPKPD.py
```

## Benchmark 2: Quadrotor Stabilization

First generate the CasADi/IPOPT reference solution:

```bash
python quadrotor_casadi.py
```

This generates the ground-truth data used for comparison by the neural-controller script.

Then train and evaluate the neural controllers:

```bash
python quadrotorNN.py
```

Run the convergence analysis:

```bash
python convergence_orderQUADROTOR.py
```

## Benchmark 3: Lorenz-96 Chaotic System

First generate the CasADi/IPOPT reference solution:

```bash
python lorenz_casadi.py
```

This generates the ground-truth data used for comparison by the neural-controller script.

Then train and evaluate the neural controllers:

```bash
python lorenzNN.py
```

Run the convergence analysis:

```bash
python convergence_orderLORENZ.py
```

## Multi-Seed Experiments

The repository also includes `run_sweep.sh` for sequential multi-seed experiments.

The runs are performed sequentially rather than concurrently so that multiple jobs do not compete for GPU resources and distort wall-clock timing measurements.

After making the script executable:

```bash
chmod +x run_sweep.sh
```

a complete sweep can be started with:

```bash
./run_sweep.sh
```

A short smoke test can be run with:

```bash
./run_sweep.sh --smoke
```

Individual benchmarks can also be selected:

```bash
./run_sweep.sh --benchmarks pkpd
```

```bash
./run_sweep.sh --benchmarks quadrotor
```

```bash
./run_sweep.sh --benchmarks lorenz
```

The number of seeds can be changed with:

```bash
./run_sweep.sh --pkpd-seeds 25
```

```bash
./run_sweep.sh --quad-seeds 25
```

```bash
./run_sweep.sh --lorenz-seeds 25
```

A dry run can be used to inspect the experiment plan without launching training:

```bash
./run_sweep.sh --dry-run
```

Completed runs are stored in the `results/` directory and execution logs are stored in the `logs/` directory.

## Training Times

The main neural-controller experiments train Euler, Midpoint, and RK4 policies for 25,000 epochs.

Approximate runtimes depend strongly on hardware.

| Script | Apple M3 Pro (Metal) | CPU |
|---|---:|---:|
| `PKPDNN.py` | ~45 min | ~150 min |
| `quadrotorNN.py` | ~3 hrs | ~10 hrs |
| `lorenzNN.py` | ~5 hrs | ~15 hrs |
| `convergence_orderPKPD.py` | ~15 min | ~45 min |
| `convergence_orderQUADROTOR.py` | ~4 hrs | ~12 hrs |
| `convergence_orderLORENZ.py` | ~6 hrs | ~18 hrs |
| `quadrotor_casadi.py` | ~2 min | ~5 min |
| `lorenz_casadi.py` | ~5 min | ~10 min |
| `analytical_pmp.py` | ~5 min | ~5 min |

These values are approximate and depend on hardware, TensorFlow configuration, and system load.

## Evaluation

Each neural-controller script evaluates three discretization methods:

- Euler
- Midpoint / RK2
- RK4

The scripts report quantities including:

- Discrete training objective
- High-fidelity policy performance
- Final-state error
- Control performance
- State trajectories
- Control trajectories
- Comparison with reference solutions

High-fidelity evaluation of the trained neural policies is performed using SciPy's adaptive RK45 solver.

### Convergence Analysis

The convergence scripts evaluate numerical error over multiple time-step sizes and compute empirical convergence behavior for:

- Euler
- Midpoint / RK2
- RK4

The analyses include:

- High-fidelity SciPy RK45 reference integration
- Cost-functional error
- State-trajectory error
- Empirical convergence-rate estimates
- Comparison with theoretical convergence orders
- Statistical tests of observed convergence behavior
- Log-log convergence plots

### Reference Solutions

The benchmark reference solutions are constructed differently for each problem:

- **PK/PD:** Pontryagin's Maximum Principle
- **Quadrotor:** CasADi/IPOPT direct transcription
- **Lorenz-96:** CasADi/IPOPT direct transcription

The CasADi scripts save benchmark trajectories and optimization results to `.npz` files for subsequent comparison with the neural controllers.

## Results

Representative results reported in the experiments are summarized below.

| Example | Quantity | Euler | Midpoint | RK4 | Reference |
|---|---|---:|---:|---:|---:|
| PK/PD | Cost | 251.47 | 251.01 | 250.96 | 250.50 |
| PK/PD | Final tumor mass (g) | 4.49 | 4.46 | 4.41 | 4.41 |
| Quadrotor | Cost | 27.84 | 27.37 | 27.32 | 27.32 |
| Quadrotor | Final position error | 0.003 | 0.006 | 0.001 | 0.000 |
| Lorenz-96 | Cost | Failed | 0.29 | 0.27 | 0.25 |
| Lorenz-96 | Final equilibrium error | Failed | 0.07 | 0.02 | 0.00 |

The numerical experiments compare the effect of the integration method used inside the neural optimal-control training loop. Higher-order integration substantially reduces discretization error across the benchmark problems, with the largest differences appearing in the more dynamically challenging systems.

## System Parameters

### PK/PD Model

| Parameter | Value |
|---|---:|
| Initial growth rate, λ₀ | 0.146 day⁻¹ |
| Exponential growth parameter, λ₁ | 0.334 day⁻¹ |
| Transit rate, k₁ | 0.469 day⁻¹ |
| Kill rate, k₂ | 8.42 × 10⁻⁴ |
| Switch exponent, ψ | 20.0 |
| Time horizon | 27 days |
| Training step size | 0.1 |
| Network | 5–3–1 MLP with SiLU |
| Convergence step sizes | 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125 |

### Quadrotor 6-DOF Model

| Parameter | Value |
|---|---:|
| Mass, m | 1.0 kg |
| Gravity, g | 9.81 m/s² |
| Inertia, Ixx | 0.01 kg·m² |
| Inertia, Iyy | 0.01 kg·m² |
| Inertia, Izz | 0.02 kg·m² |
| Thrust deviation bound, δT | ±15 N |
| Torque bound, τ | ±0.5 N·m |
| Time horizon | 5.0 s |
| Training step size | 0.02 |
| Network | 32–32–4 MLP with SiLU |
| Convergence step sizes | 0.05, 0.025, 0.0125, 0.00625, 0.003125 |

### Lorenz-96 Model

| Parameter | Value |
|---|---:|
| Dimension, N | 20 |
| Forcing, F | 8.0 |
| Control bound | ±10 |
| Target equilibrium | xᵢ = F = 8.0 |
| Time horizon | 2.0 Lorenz time units |
| Training step size | 0.01 |
| Network | 256–128–20 MLP with SiLU |
| Convergence step sizes | 0.02, 0.01, 0.005, 0.0025, 0.00125 |

## Shared Optimization Settings

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 10⁻³ |
| Epochs | 25,000 |
| Precision | Float64 |
| Activation | SiLU |
| Weight initialization | N(0, 0.05²) |

## Reproducibility

The experiments use explicit random seeds in the neural-controller and convergence-analysis scripts.

For multi-seed experiments, `run_sweep.sh` runs the benchmarks sequentially and stores individual results and logs so interrupted experiments can be resumed without rerunning completed seeds.

Because neural-network optimization and GPU execution may introduce hardware-dependent numerical variation, exact floating-point results may vary slightly across platforms.

## License

This repository is distributed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

See `LICENSE.md` for the complete license text.