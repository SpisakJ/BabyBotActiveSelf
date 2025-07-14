import torch
import matplotlib.pyplot as plt

class Pacifier:
    def __init__(self, condition="analog", dt=0.01, device="cuda"):
        self.condition = condition
        self.device = device
        self.pressure_range = (0, 1)
        self.frequency_range = (0, 400)
        self.pressure_threshold = 0.16

        # For non-analog condition
        self.trill_t0 = 0.0
        self.trill_freq = 0.0
        self.trill_duration = 0.55
        self.n_trills = 11
        self.trill_freq_range = (0, 400)

        # System states
        self.frequency = torch.tensor(0.0, device=self.device)
        self.F = torch.tensor(0.0, device=self.device)
        self.x = torch.tensor(0.0, device=self.device)
        self.v = torch.tensor(0.0, device=self.device)
        self.a = torch.tensor(0.0, device=self.device)
        self.err_int = torch.tensor(0.0, device=self.device)
        self.step = 0
        self.time = torch.tensor(0.0, device=self.device)

        # System settings
        self.m  = 0.2
        self.k  = 15.0
        self.d  = 1.7 * (self.m * self.k) ** 0.5
        self.k_I = self.k * 2.2
        self.dt = dt
        self.err_int_limit = 0.5

        # Build continuous-time state matrices for z=[x; v; I]:
        Ac = torch.tensor([
            [0, 1, 0],
            [-self.k/self.m, -self.d/self.m, self.k_I/self.m],
            [-1, 0, 0]
        ], dtype=torch.float32, device=self.device)
        Bc = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

        # Exact ZOH discretization
        self.Ad = torch.matrix_exp(Ac * self.dt)
        self.Bd = torch.linalg.solve(Ac, (self.Ad - torch.eye(3, device=self.device))) @ Bc

        # Pack state
        self.state = torch.zeros(3, dtype=torch.float32, device=self.device)   # [x, v, I]

        # Data logging
        self.x_log = []
        self.F_log = []
        self.x_desired_log = []
        self.frequency_log = []
        self.time_log = []

    def step_mass_spring_damper(self, x_desired):
        # x_desired: tensor, requires_grad as needed
        z = self.Ad @ self.state + self.Bd * x_desired
        self.state = z
        self.x, self.v, self.err_int = z
        self.state = self.state.detach()
        # compute force for logging
        self.F = self.k_I * self.err_int
        self.time = self.time + self.dt

        # log data
        self.x_log.append(self.x)
        self.F_log.append(self.F)
        self.x_desired_log.append(x_desired)
        self.time_log.append(self.time)

    def map_pressure_to_frequency(self, pressure):
        if self.condition == "analog":
            self.map_pressure_to_frequency_proportional(pressure)
        elif self.condition == "non-analog":
            self.map_pressure_to_frequency_rand(pressure)
        else:
            raise ValueError(f"Invalid condition '{self.condition}'.")

    def map_pressure_to_frequency_rand(self, pressure):
        # This function is not differentiable due to randomness and indexing
        # For differentiable use, you may want to skip or replace this logic
        time_diff = self.time.item() - self.trill_t0
        if time_diff > self.trill_duration or self.trill_t0 == 0.0:
            if pressure.item() > self.pressure_threshold:
                self.trill_t0 = self.time.item()
                self.trill_freq = torch.rand(self.n_trills, device=self.device) * (self.trill_freq_range[1] - self.trill_freq_range[0]) + self.trill_freq_range[0]
                self.frequency = self.trill_freq[0]
            else:
                self.frequency = torch.tensor(0.0, device=self.device)
        else:
            frequ_index = int(time_diff // (self.trill_duration/self.n_trills))
            if frequ_index >= self.n_trills:
                frequ_index = self.n_trills - 1
            self.frequency = self.trill_freq[frequ_index]
        self.frequency_log.append(self.frequency)

    def map_pressure_to_frequency_proportional(self, pressure):
        if pressure > self.pressure_threshold:
            pressure_min, pressure_max = self.pressure_range
            frequency_min, frequency_max = self.frequency_range
            pressure = torch.clamp(pressure, pressure_min, pressure_max)
            self.frequency = ((pressure - pressure_min) / (pressure_max - pressure_min)) * (frequency_max - frequency_min) + frequency_min
        else:
            self.frequency = torch.tensor(0.0, device=self.device)
        self.frequency_log.append(self.frequency)

    def run(self, desired_pressure, steps=1):
        """
        desired_pressure: torch tensor, can require grad
        Returns: final x (tensor, allows backprop to desired_pressure)
        """
        for _ in range(steps):
            self.step_mass_spring_damper(desired_pressure)
            self.map_pressure_to_frequency(self.x)
        return self.x

    def visualize_system(self, show_pitch=False):
        fig, axs = plt.subplots(1, 1, figsize=(5, 3))
        time_points = torch.arange(len(self.x_log)) * self.dt

        axs.plot(time_points.cpu(), torch.stack(self.x_log).cpu(), label=r'$x$', color="blue")
        axs.plot(time_points.cpu(), torch.stack(self.x_desired_log).cpu(), label=r'$x_{\rm des}$', color="orange")
        if show_pitch:
            axs.plot(time_points.cpu(), torch.stack(self.frequency_log).cpu(), label=r'$p$', color="green")
        axs.axhline(y=0.1, color='r', linestyle='--', label=r'$x_{\rm thr}$')
        axs.set_ylabel('Pressure (psi)')
        axs.legend()
        axs.grid(True)
        axs.set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pac_env = Pacifier(condition="analog", dt=0.01, device=device)

    # Example: differentiable run
    duration = 1.0
    steps = int(duration / pac_env.dt)
    desired_pressure = torch.tensor(0.5, dtype=torch.float32, device=device, requires_grad=True)
    final_x = pac_env.run(desired_pressure, steps=steps)

    # Example backward
    loss = (final_x - 0.3) ** 2
    loss.backward()
    print("Gradient w.r.t. desired_pressure:", desired_pressure.grad)

    pac_env.visualize_system()