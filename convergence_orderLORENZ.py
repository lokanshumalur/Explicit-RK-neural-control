import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci_int
import time
from scipy import stats
import random

# 0. SETUP
RANDOM_SEED = 26
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
if tf.keras.backend.floatx() != 'float64':
    tf.keras.backend.set_floatx('float64')

DTYPE_TF = tf.keras.backend.floatx()
DTYPE_NP = np.float64
print(f"INFO: TensorFlow precision set to {DTYPE_TF}.")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    pass

# 1. BUTCHER TABLEAUS
BUTCHER_TABLEAUS = {
    "euler": {
        "A": np.array([[0.0]], dtype=np.float64),
        "b": np.array([1.0], dtype=np.float64),
        "c": np.array([0.0], dtype=np.float64)
    },
    "midpoint": {
        "A": np.array([[0.0, 0.0],
                       [0.5, 0.0]], dtype=np.float64),
        "b": np.array([0.0, 1.0], dtype=np.float64),
        "c": np.array([0.0, 0.5], dtype=np.float64)
    },
    "rk4": {
        "A": np.array([[0, 0, 0, 0],
                       [0.5, 0, 0, 0],
                       [0, 0.5, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float64),
        "b": np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=np.float64),
        "c": np.array([0, 0.5, 0.5, 1], dtype=np.float64)
    }
}

# 2. MLP
class MLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_layers, activations, output_dim=None, u_max=None):
        super(MLP, self).__init__()
        self.output_dim = input_dim if output_dim is None else output_dim
        self.input_dim = input_dim

        if u_max is not None:
            u_max_np = np.atleast_1d(np.asarray(u_max, dtype=np.float64))
            if u_max_np.size == 1 and (output_dim is not None and output_dim == 1):
                self.u_max = tf.constant(u_max_np.item(), dtype=tf.float64)
                self._bound_mode = "sigmoid"
            else:
                self.u_max = tf.constant(u_max_np.reshape(1, -1), dtype=tf.float64)
                self._bound_mode = "tanh"
        else:
            self.u_max = None
            self._bound_mode = None

        self.hidden_layers = []
        for units, act in zip(hidden_layers, activations):
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units,
                    activation=getattr(tf.nn, act),
                    dtype=tf.float64,
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                    bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
                )
            )
        self.output_layer = tf.keras.layers.Dense(
            self.output_dim,
            activation=None,
            dtype=tf.float64,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        )

    def call(self, state_t_x):
        x = state_t_x
        for layer in self.hidden_layers:
            x = layer(x)
        raw_output = self.output_layer(x)

        if self._bound_mode == "sigmoid":
            return self.u_max * tf.nn.sigmoid(raw_output)
        elif self._bound_mode == "tanh":
            return self.u_max * tf.nn.tanh(raw_output)
        return raw_output

# 3. LORENZ-96 CHAOTIC DYNAMICS (20D)
class Lorenz96Dynamics:
    def __init__(self, N=20, F=8.0):
        self.N = N
        self.F = F

    def __call__(self, x, u, t, g=None):
        """
        x : [batch, N]   physical state
        u : [batch, N]   control forcing
        Returns: [batch, N]  dx/dt
        """
        x_ip1 = tf.roll(x, shift=-1, axis=1)    # x_{i+1}
        x_im1 = tf.roll(x, shift=1, axis=1)     # x_{i-1}
        x_im2 = tf.roll(x, shift=2, axis=1)     # x_{i-2}

        # Eq. 5: dx_i/dt = (x_{i+1} − x_{i-2}) x_{i-1} − x_i + F + u_i
        dx = (x_ip1 - x_im2) * x_im1 - x + self.F + u

        return dx

# 4. Costs — with reference state tracking
class RunningCost:
    def __init__(self, Q, R, x_ref=None):
        self.Q = tf.constant(Q, dtype=tf.float64)
        self.R = tf.constant(R, dtype=tf.float64)
        self.x_ref = tf.constant(x_ref, dtype=tf.float64) if x_ref is not None else None

    def __call__(self, x, u, t=None):
        e = (x - self.x_ref) if self.x_ref is not None else x
        state_cost = tf.reduce_sum((e @ self.Q) * e, axis=1, keepdims=True)
        control_cost = tf.reduce_sum((u @ self.R) * u, axis=1, keepdims=True)
        return state_cost + control_cost

class TerminalCost:
    def __init__(self, S, x_ref=None):
        self.S = tf.constant(S, dtype=tf.float64)
        self.x_ref = tf.constant(x_ref, dtype=tf.float64) if x_ref is not None else None

    def __call__(self, x, u=None):
        e = (x - self.x_ref) if self.x_ref is not None else x
        return tf.reduce_sum((e @ self.S) * e, axis=1, keepdims=True)

# 5. Trajectory Simulator
class TrajectorySimulator:
    def __init__(self, dynamics_fn, model, running_cost_fn, terminal_cost_fn, butcher_tableau, u_max):
        self.dynamics_fn = dynamics_fn
        self.model = model
        self.running_cost = running_cost_fn
        self.terminal_cost = terminal_cost_fn
        self.butcher_tableau = butcher_tableau
        self.u_max = u_max

    def simulate_trajectory(self, x0, dt, T_final, track_trajectory=False):
        steps = int(T_final / dt)
        x = tf.convert_to_tensor(x0, dtype=tf.float64)
        total_cost = tf.constant([[0.0]], dtype=tf.float64)

        trajectory_x = [] if track_trajectory else None
        trajectory_u = [] if track_trajectory else None

        A, b, c = tf.constant(self.butcher_tableau["A"], dtype=DTYPE_TF), \
            tf.constant(self.butcher_tableau["b"], dtype=DTYPE_TF), \
            tf.constant(self.butcher_tableau["c"], dtype=DTYPE_TF)
        s = A.shape[0]

        dt_f = tf.constant(dt, dtype=DTYPE_TF)
        T_final_f = tf.constant(T_final, dtype=DTYPE_TF)

        for step in tf.range(steps):
            t_current = tf.cast(step, dtype=DTYPE_TF) * dt_f

            k_values, l_values, u_values = [], [], []

            tau_start = t_current / T_final_f
            state_start = tf.concat([tf.reshape(tau_start, [1, 1]), x], axis=1)
            u_start = self.model(state_start)

            if track_trajectory:
                trajectory_x.append(x)
                trajectory_u.append(u_start)

            for i in range(s):
                x_k_arg = x
                for j in range(i):
                    x_k_arg += dt_f * A[i, j] * k_values[j]

                t_k_arg = t_current + dt_f * c[i]
                tau_k = t_k_arg / T_final_f

                state_k = tf.concat([tf.reshape(tau_k, [1, 1]), x_k_arg], axis=1)
                u_k = self.model(state_k)

                k_i = self.dynamics_fn(x_k_arg, u_k, t_k_arg, None)
                k_values.append(k_i)
                l_values.append(self.running_cost(x_k_arg, u_k))

            x_next = x
            for i in range(s):
                x_next += dt_f * b[i] * k_values[i]
            x = x_next

            step_cost = tf.constant([[0.0]], dtype=tf.float64)
            for i in range(s):
                step_cost += b[i] * l_values[i]
            total_cost += dt_f * step_cost

        total_cost += self.terminal_cost(x)

        if track_trajectory:
            trajectory_x.append(x)
            trajectory_u.append(trajectory_u[-1] if trajectory_u else tf.constant([[0.0]], dtype=DTYPE_TF))

            final_trajectory_x = np.concatenate([t.numpy() for t in trajectory_x], axis=0)
            final_trajectory_u = np.concatenate([u.numpy() for u in trajectory_u], axis=0)
            time_points = np.linspace(0, T_final, len(trajectory_x))

            final_trajectory = {
                't': time_points,
                'x': final_trajectory_x,
                'u': final_trajectory_u
            }
        else:
            final_trajectory = None

        return x, tf.squeeze(total_cost), final_trajectory

# 6. Optimal Controller
class OptimalController:
    def __init__(self, dynamics_fn, running_cost_fn, terminal_cost_fn, initial_state,
                 hidden_units, activations, input_dim, output_dim, dt, T_final, butcher_tableau, u_max,
                 t_treatment_start, optimizer=None):

        self.dynamics_fn = dynamics_fn
        self.running_cost = running_cost_fn
        self.terminal_cost = terminal_cost_fn
        self.butcher_tableau_name = butcher_tableau

        self.model = MLP(input_dim=input_dim, hidden_layers=hidden_units,
                         activations=activations, output_dim=output_dim, u_max=u_max)

        self.simulator = TrajectorySimulator(
            dynamics_fn, self.model, running_cost_fn, terminal_cost_fn, BUTCHER_TABLEAUS[butcher_tableau], u_max
        )

        self.x0 = tf.convert_to_tensor(initial_state, dtype=tf.float64)
        self.dt = dt
        self.T_final = T_final
        self.t_treatment_start = t_treatment_start

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        else:
            self.optimizer = optimizer

        self.loss_history = []
        self.trajectory = None
        self.u_max = u_max

    @tf.function
    def training_step(self):
        with tf.GradientTape() as tape:
            _, loss_val, _ = self.simulator.simulate_trajectory(self.x0, self.dt, self.T_final, track_trajectory=False)
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_val

    def train(self, epochs=500):
        start_time = time.time()
        print(f"\n=== Training started with {self.butcher_tableau_name.upper()}, dt={self.dt} for {epochs} epochs ===")
        for epoch in range(1, epochs + 1):
            loss_val = self.training_step()
            self.loss_history.append(loss_val.numpy())

            if epoch % max(1, epochs // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{epochs}, Loss={loss_val.numpy():.6f}, Time={elapsed:.1f}s")
                start_time = time.time()

        _, final_J, self.trajectory = self.simulator.simulate_trajectory(self.x0, self.dt, self.T_final,
                                                                         track_trajectory=True)
        print(f"\nTraining Complete. Final J_disc = {final_J.numpy():.6f}")
        return self.loss_history

# 7. HIGH-FIDELITY REFERENCE GENERATOR
def scipy_reference_generator(model, dynamics_fn, running_cost, terminal_cost, x0_np, T_final):
    def augmented_ode_func(t, Y):
        x_np = Y[:-1]
        tau = tf.constant([[t / T_final]], dtype=DTYPE_TF)
        x_tf = tf.convert_to_tensor(x_np.reshape(1, -1), dtype=DTYPE_TF)
        state_t_x = tf.concat([tau, x_tf], axis=1)

        u_tf = model(state_t_x)
        x_dot_tf = dynamics_fn(x_tf, u_tf, tf.constant([[t]], dtype=DTYPE_TF), None)
        x_dot_np = x_dot_tf.numpy().flatten()

        L_tf = running_cost(x_tf, u_tf)
        L_np = L_tf.numpy().item()

        return np.append(x_dot_np, L_np)

    Y0 = np.append(x0_np.flatten(), 0.0)

    sol = sci_int.solve_ivp(
        augmented_ode_func,
        [0, T_final],
        Y0,
        method='RK45',
        rtol=1e-10,
        atol=1e-13,
        dense_output=True
    )

    x_final_np = sol.y[:-1, -1]
    J_running_final = sol.y[-1, -1]

    x_final_tf = tf.convert_to_tensor(x_final_np.reshape(1, -1), dtype=DTYPE_TF)
    J_terminal = terminal_cost(x_final_tf).numpy().item()
    J_ref = J_running_final + J_terminal

    return J_ref, sol

# 8. CONVERGENCE ANALYSIS
def analyze_convergence(model, dynamics_fn, running_cost, terminal_cost, x0_np, T_final,
                        butcher_method, step_sizes, J_ref, sol_ref, u_max):
    dt_values = []
    cost_errors = []
    state_errors = []

    print(f"\n=== Evaluating Policy with {butcher_method.upper()} Stepper ===")

    for dt in step_sizes:
        simulator = TrajectorySimulator(
            dynamics_fn, model, running_cost, terminal_cost, BUTCHER_TABLEAUS[butcher_method], u_max
        )

        _, J_val, trajectory = simulator.simulate_trajectory(
            x0_np, dt, T_final, track_trajectory=True
        )

        J_val_scalar = J_val.numpy()
        cost_error = np.abs(J_val_scalar - J_ref)

        t_points = trajectory['t']
        traj_x = trajectory['x']
        traj_ref = sol_ref.sol(t_points)[:-1, :].T
        state_error = np.sqrt(np.mean((traj_x - traj_ref) ** 2))

        dt_values.append(dt)
        cost_errors.append(cost_error)
        state_errors.append(state_error)

        print(f"  dt = {dt:.6f}, Cost Error = {cost_error:.6e}, State RMS Error = {state_error:.6e}")

    return np.array(dt_values), np.array(cost_errors), np.array(state_errors)

# 9. MAIN EXECUTION
if __name__ == "__main__":
    # ── Lorenz-96 parameters ──
    N_LORENZ = 20
    F_LORENZ = 8.0    # chaotic regime

    dynamics = Lorenz96Dynamics(N=N_LORENZ, F=F_LORENZ)

    # ── Reference state: the unstable equilibrium x_i* = F ──
    x_ref = F_LORENZ * np.ones((1, N_LORENZ))

    # ── Network architecture ──
    STATE_DIM   = N_LORENZ   # 20
    CONTROL_DIM = N_LORENZ   # 20
    hidden_units = [256, 128]
    activations  = ["silu", "silu"]
    input_dim    = 1 + STATE_DIM   # 21
    output_dim   = CONTROL_DIM     # 20

    # Control bounds (symmetric via tanh): u_i ∈ [−10, 10]
    u_max = 10.0 * np.ones(CONTROL_DIM)

    # ── Cost matrices ──
    Q_val = 1.0 * np.eye(STATE_DIM)
    R_val = 0.01 * np.eye(CONTROL_DIM)
    S_val = 5.0 * np.eye(STATE_DIM)

    running_cost  = RunningCost(Q=Q_val, R=R_val, x_ref=x_ref)
    terminal_cost = TerminalCost(S=S_val, x_ref=x_ref)

    # ── Initial condition: perturbed from equilibrium ──
    perturbation = 0.5 * np.sin(2.0 * np.pi * np.arange(N_LORENZ) / N_LORENZ)
    x0_lorenz = [(F_LORENZ + perturbation).tolist()]

    T_final = 2.0
    t_treatment_start = 0

    # Train and fix the policy
    training_method = "rk4"
    training_dt = 0.01
    training_epochs = 25000

    controller = OptimalController(
        dynamics_fn=dynamics, running_cost_fn=running_cost, terminal_cost_fn=terminal_cost,
        initial_state=x0_lorenz, hidden_units=hidden_units, activations=activations,
        input_dim=input_dim, output_dim=output_dim, dt=training_dt, T_final=T_final,
        butcher_tableau=training_method, u_max=u_max, t_treatment_start=t_treatment_start
    )
    controller.train(epochs=training_epochs)

    # High-fidelity reference
    print("\n" + "=" * 80)
    print("COMPUTING HIGH-FIDELITY REFERENCE (SciPy RK45)")
    print("=" * 80)
    J_ref, sol_ref = scipy_reference_generator(
        controller.model, dynamics, running_cost, terminal_cost, controller.x0.numpy(), T_final
    )
    print(f"Reference J_ref = {J_ref:.15f} (SciPy RK45, atol=1e-13)")

    # Convergence analysis
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    step_sizes = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125])
    methods_to_test = ["euler", "midpoint", "rk4"]
    results = {}

    for method in methods_to_test:
        dt_vals, cost_errors, state_errors = analyze_convergence(
            controller.model, dynamics, running_cost, terminal_cost,
            controller.x0.numpy(), T_final, method, step_sizes, J_ref, sol_ref, u_max=u_max
        )
        results[method] = {
            'dt': dt_vals,
            'cost_errors': cost_errors,
            'state_errors': state_errors
        }

    # Plot convergence
    print("\n" + "=" * 80)
    print("PLOTTING CONVERGENCE")
    print("=" * 80)

    colors = {'euler': '#E63946', 'midpoint': '#457B9D', 'rk4': '#2A9D8F'}
    markers = {'euler': 'o', 'midpoint': 'D', 'rk4': 's'}
    dt_span_for_slopes = np.array([step_sizes[-1] * 0.9, step_sizes[0] * 1.1])

    def plot_reference_slope(ax, dt_data, error_data, order, color, style, label):
        base_dt = dt_data[1]
        base_error = error_data[1]
        error_ref = base_error * (dt_span_for_slopes / base_dt) ** order
        ax.loglog(dt_span_for_slopes, error_ref, color=color, linestyle=style, alpha=0.7, linewidth=2, label=label)

    # Plot 1: Cost Error
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    for method in methods_to_test:
        label_map = {'euler': r'$\mathit{Euler}$', 'midpoint': r'$\mathit{Midpoint}$', 'rk4': r'$\mathit{RK4}$'}
        label_text = label_map[method]
        ax1.loglog(results[method]['dt'], results[method]['cost_errors'], marker=markers[method], color=colors[method],
                   label=label_text, linewidth=0, markersize=10, markeredgecolor='black', alpha=0.9)
        ax1.loglog(results[method]['dt'], results[method]['cost_errors'], color=colors[method], linewidth=3, alpha=0.5)

    plot_reference_slope(ax1, results['euler']['dt'], results['euler']['cost_errors'], 1, 'k', '--',
                         r'$\mathit{Theoretical\ O(\Delta t^{1})}$')
    plot_reference_slope(ax1, results['midpoint']['dt'], results['midpoint']['cost_errors'], 2, 'k', '-.',
                         r'$\mathit{Theoretical\ O(\Delta t^{2})}$')
    plot_reference_slope(ax1, results['rk4']['dt'], results['rk4']['cost_errors'], 4, 'k', ':',
                         r'$\mathit{Theoretical\ O(\Delta t^{4})}$')

    ax1.set_xlabel(r'Step Size ($\Delta t$)', fontsize=16)
    ax1.set_ylabel(r'Absolute Error', fontsize=16)
    ax1.set_title('Cost convergence analysis', fontsize=18, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.invert_xaxis()
    plt.tight_layout()
    plt.show()

    # Plot 2: State Trajectory Error
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    for method in methods_to_test:
        label_map = {'euler': r'$\mathit{Euler}$', 'midpoint': r'$\mathit{Midpoint}$', 'rk4': r'$\mathit{RK4}$'}
        label_text = label_map[method]
        ax2.loglog(results[method]['dt'], results[method]['state_errors'], marker=markers[method], color=colors[method],
                   label=label_text, linewidth=0, markersize=10, markeredgecolor='black', alpha=0.9)
        ax2.loglog(results[method]['dt'], results[method]['state_errors'], color=colors[method], linewidth=3, alpha=0.5)

    plot_reference_slope(ax2, results['euler']['dt'], results['euler']['state_errors'], 1, 'k', '--',
                         r'$\mathit{Theoretical\ O(\Delta t^{1})}$')
    plot_reference_slope(ax2, results['midpoint']['dt'], results['midpoint']['state_errors'], 2, 'k', '-.',
                         r'$\mathit{Theoretical\ O(\Delta t^{2})}$')
    plot_reference_slope(ax2, results['rk4']['dt'], results['rk4']['state_errors'], 4, 'k', ':',
                         r'$\mathit{Theoretical\ O(\Delta t^{4})}$')

    ax2.set_xlabel(r'Step Size ($\Delta t$)', fontsize=16)
    ax2.set_ylabel('RMS State Trajectory Error', fontsize=16)
    ax2.set_title('State trajectory convergence analysis', fontsize=18, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.invert_xaxis()
    plt.tight_layout()
    plt.show()

    # Empirical convergence rates
    print("\n" + "=" * 80)
    print("EMPIRICAL CONVERGENCE RATES")
    print("=" * 80)
    final_J_disc = controller.simulator.simulate_trajectory(controller.x0, controller.dt, controller.T_final)[1].numpy()
    print(f"Policy Trained J_disc = {final_J_disc:.6f}")
    print(f"Policy True J_ref   = {J_ref:.6f}")

    for method in methods_to_test:
        dt_vals = results[method]['dt']
        cost_errors = results[method]['cost_errors']
        state_errors = results[method]['state_errors']
        log_dt = np.log(dt_vals)

        coeffs_cost = np.polyfit(log_dt, np.log(cost_errors), 1)
        coeffs_state = np.polyfit(log_dt, np.log(state_errors), 1)

        empirical_order_cost = coeffs_cost[0]
        empirical_order_state = coeffs_state[0]

        theoretical = {'euler': 1.0, 'midpoint': 2.0, 'rk4': 4.0}[method]
        print(f"\n{method.upper()}:")
        print(f"  Cost Error Order: {empirical_order_cost:.3f} (Theoretical: {theoretical})")
        print(f"  State RMS Error Order: {empirical_order_state:.3f} (Theoretical: {theoretical})")

    # Statistical testing
    print("\n" + "=" * 80)
    print("STATISTICAL TESTING (ONE-SIDED T-TEST)")
    print("=" * 80)

    def regression_ttest(h, err, mu0):
        x = np.log(h)
        y = np.log(err)
        n = len(x)
        slope, intercept, r_value, p_two_sided, std_err = stats.linregress(x, y)
        t_stat = (slope - mu0) / std_err
        df = n - 2
        p_one_sided = stats.t.cdf(t_stat, df)
        return {
            "slope": slope,
            "intercept": intercept,
            "std_err": std_err,
            "t_statistic": t_stat,
            "p_value_one_sided": p_one_sided,
            "df": df,
            "r_squared": r_value ** 2
        }

    euler_mu0 = 1.0
    midpoint_mu0 = 2.0
    rk4_mu0 = 4.0

    h = results['euler']['dt']
    cost_euler = results['euler']['cost_errors']
    state_euler = results['euler']['state_errors']
    cost_midpoint = results['midpoint']['cost_errors']
    state_midpoint = results['midpoint']['state_errors']
    cost_rk4 = results['rk4']['cost_errors']
    state_rk4 = results['rk4']['state_errors']

    res_cost_euler = regression_ttest(h, cost_euler, euler_mu0)
    res_state_euler = regression_ttest(h, state_euler, euler_mu0)
    res_cost_midpoint = regression_ttest(h, cost_midpoint, midpoint_mu0)
    res_state_midpoint = regression_ttest(h, state_midpoint, midpoint_mu0)
    res_cost_rk4 = regression_ttest(h, cost_rk4, rk4_mu0)
    res_state_rk4 = regression_ttest(h, state_rk4, rk4_mu0)

    def report(name, res):
        print(name)
        print(f"  slope estimate beta-hat = {res['slope']:.6f}")
        print(f"  intercept               = {res['intercept']:.6f}")
        print(f"  std. error              = {res['std_err']:.6f}")
        print(f"  t statistic             = {res['t_statistic']:.6f}")
        print(f"  p-value (1-sided)       = {res['p_value_one_sided']:.6e}")
        print(f"  df                      = {res['df']}")
        print(f"  R^2                     = {res['r_squared']:.6f}")
        print()

    report("Euler (Cost Error)", res_cost_euler)
    report("Euler (State Error)", res_state_euler)
    report("Midpoint (Cost Error)", res_cost_midpoint)
    report("Midpoint (State Error)", res_state_midpoint)
    report("RK4 (Cost Error)", res_cost_rk4)
    report("RK4 (State Error)", res_state_rk4)