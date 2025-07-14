import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class PacifierV4:
    """Implementation from Pressure2Soundv4_usedInFirstPaper.py"""
    def __init__(self, condition="analog"):
        
        self.condition = condition
        
        # for analog condition
        self.pressure_range = (0, 1)
        self.frequency_range = (0, 400)
        self.pressure_threshold = 0.1  # Pressure threshold for producing sound

        # for non-analog condition
        self.trill_t0 = 0.0
        self.trill_freq = 0.0
        self.trill_duration = 0.55
        self.n_trills = 11
        self.trill_freq_range = (0, 400)

        # system states
        self.frequency = 0.0
        self.F = 0.0
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0
        self.err_int = 0.0
        self.step = 0
        self.time = 0.0

        # system settings
        self.m = 1
        self.k = 15
        self.d = 1.5*np.sqrt(self.m*self.k)
        self.equ_noise = 0
        self.x_desired_noise = 0
        self.F_noise = 0
        self.dt = 0.001

        # data logging
        self.x_log = np.array([self.x])
        self.F_log = np.array([self.F])
        self.x_desired_log = np.array([0.0])
        self.frequency_log = np.array([self.frequency])
        self.time_log = np.array([self.time])

    def step_mass_spring_damper(self, x_desired):
        """
            Mass spring damper system with I control to model the dynamics of the pressure in the mouth.

            Parameters
            ----------
            x_desired : float
                Desired position (= desired pressure) as input to the dynamical system 
        """

        if x_desired != self.x_desired_log[-1]:
            self.x_desired_noise = np.random.normal(0, 0.01, 1)
            self.equ_noise = np.random.normal(-0.01, 0.01, 1)
        
        if self.F_noise == 0:
            self.F_noise = np.random.normal(0.006,0.006/2,1)
        
        x_target = x_desired + self.x_desired_noise
        
        err = (x_target-self.x)
        self.err_int = self.err_int + err

        if x_target > 0.05:
            # k_I = (np.tanh(80*err)+1)*(0.008+self.F_noise)
            k_I = (np.tanh(80*abs(err))+1)*(0.008+self.F_noise) * (self.dt/0.001)
            F_u = k_I*self.err_int
            # k_I = (np.tanh(80*err)+1)*(0.008) * (self.dt/0.001)
            # F_u = k_I*self.err_int+self.F_noise*200

            F = self.F + 1*(F_u-self.F)
        else:
            F_u = 0
            self.F_noise = 0
            self.err_int = 0
            F = self.F + 1*(F_u-self.F)

        F = max(min(F, 15), -15)

        # perform one step & update states
        self.F = F
        self.a = (F + self.d*(0-self.v) + self.k*(self.equ_noise-self.x)) / self.m
        self.v = self.v + self.a*self.dt
        self.x = self.x + self.v*self.dt
        self.step = self.step +1
        self.time = self.step*self.dt

        # log data
        self.x_log = np.append(self.x_log, self.x)
        self.F_log = np.append(self.F_log, self.F)
        self.x_desired_log = np.append(self.x_desired_log, x_desired)
        self.time_log = np.append(self.time_log, self.time)
    
    def map_pressure_to_frequency(self, pressure):
        """
            Selecting the mapping function for mapping pressure to frequency

            Parameters
            ----------
            pressure : float
                Pressure for which to perform the mapping
        """
        
        if self.condition == "analog":
            self.map_pressure_to_frequency_proportional(pressure)
        elif self.condition == "non-analog":
            self.map_pressure_to_frequency_rand(pressure)
        else:
            raise ValueError(f"Invalid condition '{self.condition}'. Valid conditions are 'analog' and 'non-analog'.")
    
    def map_pressure_to_frequency_rand(self, pressure):
        """
            Mapping function for the non-analog condition

            Parameters
            ----------
            pressure : float
                Pressure for which to perform the mapping
        """

        time_diff = self.time - self.trill_t0
        if time_diff > self.trill_duration or self.trill_t0 == 0.0:
            if pressure > self.pressure_threshold:
                self.trill_t0 = self.time
                self.trill_freq = np.random.uniform(self.trill_freq_range[0], self.trill_freq_range[1], self.n_trills)
                self.frequency = self.trill_freq[0]
            else:
                self.frequency = 0
        else:
            frequ_index = int(time_diff // (self.trill_duration/self.n_trills))

            # Ensure the index is within the bounds of the array
            if frequ_index >= self.n_trills:
                frequ_index = self.n_trills-1

            # Select the frequency
            self.frequency = self.trill_freq[frequ_index]
        
        self.frequency_log = np.append(self.frequency_log, self.frequency)
    
    def map_pressure_to_frequency_proportional(self, pressure):
        """
            Mapping function for the analog condition

            Parameters
            ----------
            pressure : float
                Pressure for which to perform the mapping
        """

        if pressure > self.pressure_threshold:
            pressure_min, pressure_max = self.pressure_range
            frequency_min, frequency_max = self.frequency_range
            pressure = max(min(pressure, pressure_max), pressure_min)  # Clamp pressure within range
            self.frequency = ((pressure - pressure_min) / (pressure_max - pressure_min)) * (frequency_max - frequency_min) + frequency_min
        else:
            self.frequency = 0.0
            
        self.frequency_log = np.append(self.frequency_log, self.frequency)
    
    def run(self, desired_pressure, steps=1):
        """
            Simulate the pacifier using a mass sping damper system and the analog/non-analog condition.

            Parameters
            ----------
            desired_pressure : float
                Desired pressure as input to the system
            steps : int, optional
                The number of steps to perform the simulation for
        """

        for iter in range(steps):
            self.step_mass_spring_damper(x_desired=desired_pressure)
            self.map_pressure_to_frequency(pressure=self.x)


class PacifierV6:
    """Implementation from Pressure2Soundv6.py"""
    def __init__(self, condition="analog", dt=0.01):
        
        self.condition = condition
        
        # for analog condition
        self.pressure_range = (0, 1)
        self.frequency_range = (0, 400)
        self.pressure_threshold = 0.1  # Pressure threshold for producing sound

        # for non-analog condition
        self.trill_t0 = 0.0
        self.trill_freq = 0.0
        self.trill_duration = 0.55
        self.n_trills = 11
        self.trill_freq_range = (0, 400)

        # system states
        self.frequency = 0.0
        self.F = 0.0
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0
        self.err_int = 0.0
        self.step = 0
        self.time = 0.0

        # system settings
        self.m  = 0.2 #0.2
        self.k  = 15.0 #15.0
        self.d  = 1.7*np.sqrt(self.m*self.k)
        self.k_I= self.k*1.7 #*1.7#*2.2
        self.dt = dt
        self.err_int_limit = 0.5

        # Stability proof:
        # 1. Continuous-time characteristic polynomial:
        #      m*s^3 + d*s^2 + k*s + k_I = 0
        #    Routh–Hurwitz condition: d>0, k>0, k_I>0, and d*k > m*k_I
        #    Here: d*k = 2.9444*15 = 44.166 > m*k_I = 0.2*33.0 = 6.6  ✓
        #    ⇒ all continuous poles have Re(s)<0
        # 2. Exact ZOH discretization yields eigenvalues λ_i = exp(s_i*dt)
        #    ⇒ |λ_i| = exp(Re(s_i)*dt) < 1 for any dt>0
        #    ⇒ discrete-time map is a strict contraction, globally stable

        # build continuous-time state matrices for z=[x; v; I]:
        #   x' = v
        #   v' = (−k x − d v + k_I I)/m
        #   I' = (x_des−x)
        Ac = np.array([
            [   0,        1,       0      ],
            [ -self.k/self.m, -self.d/self.m, self.k_I/self.m ],
            [  -1,        0,       0      ],
        ])
        Bc = np.array([0.0, 0.0, 1.0])

        # exact ZOH discretization
        self.Ad = expm(Ac * self.dt)
        # Bd = ∫₀ᵈᵗ e^{Ac τ} dτ · Bc = Ac^{-1}(Ad − I) Bc
        self.Bd = np.linalg.solve(Ac, (self.Ad - np.eye(3))) @ Bc

        # pack state
        self.state = np.zeros(3)   # [x, v, I]

        # data logging
        self.x_log = np.array([self.x])
        self.F_log = np.array([self.F])
        self.x_desired_log = np.array([0.0])
        self.frequency_log = np.array([self.frequency])
        self.time_log = np.array([self.time])

    def step_mass_spring_damper(self, x_desired):
        """
            Mass spring damper system with I control to model the dynamics of the pressure in the mouth.

            Parameters
            ----------
            x_desired : float
                Desired position (= desired pressure) as input to the dynamical system 
        """

        # exact discrete update of [x;v;I]
        z = self.Ad @ self.state + self.Bd * x_desired
        self.state = z
        self.x, self.v, self.err_int = z

        # compute force for logging
        self.F = self.k_I * self.err_int
        self.time += self.dt

        # log data
        self.x_log = np.append(self.x_log, self.x)
        self.F_log = np.append(self.F_log, self.F)
        self.x_desired_log = np.append(self.x_desired_log, x_desired)
        self.time_log = np.append(self.time_log, self.time)
    
    def map_pressure_to_frequency(self, pressure):
        """
            Selecting the mapping function for mapping pressure to frequency

            Parameters
            ----------
            pressure : float
                Pressure for which to perform the mapping
        """
        
        if self.condition == "analog":
            self.map_pressure_to_frequency_proportional(pressure)
        elif self.condition == "non-analog":
            self.map_pressure_to_frequency_rand(pressure)
        else:
            raise ValueError(f"Invalid condition '{self.condition}'. Valid conditions are 'analog' and 'non-analog'.")
    
    def map_pressure_to_frequency_rand(self, pressure):
        """
            Mapping function for the non-analog condition

            Parameters
            ----------
            pressure : float
                Pressure for which to perform the mapping
        """

        time_diff = self.time - self.trill_t0
        if time_diff > self.trill_duration or self.trill_t0 == 0.0:
            if pressure > self.pressure_threshold:
                self.trill_t0 = self.time
                self.trill_freq = np.random.uniform(self.trill_freq_range[0], self.trill_freq_range[1], self.n_trills)
                self.frequency = self.trill_freq[0]
            else:
                self.frequency = 0
        else:
            frequ_index = int(time_diff // (self.trill_duration/self.n_trills))

            # Ensure the index is within the bounds of the array
            if frequ_index >= self.n_trills:
                frequ_index = self.n_trills - 1

            # Select the frequency
            self.frequency = self.trill_freq[frequ_index]
        
        self.frequency_log = np.append(self.frequency_log, self.frequency)
    
    def map_pressure_to_frequency_proportional(self, pressure):
        """
            Mapping function for the analog condition

            Parameters
            ----------
            pressure : float
                Pressure for which to perform the mapping
        """

        if pressure > self.pressure_threshold:
            pressure_min, pressure_max = self.pressure_range
            frequency_min, frequency_max = self.frequency_range
            pressure = max(min(pressure, pressure_max), pressure_min)  # Clamp pressure within range
            self.frequency = ((pressure - pressure_min) / (pressure_max - pressure_min)) * (frequency_max - frequency_min) + frequency_min
        else:
            self.frequency = 0.0
            
        self.frequency_log = np.append(self.frequency_log, self.frequency)
    
    def run(self, desired_pressure, steps=1):
        """
            Simulate the pacifier using a mass sping damper system and the analog/non-analog condition.

            Parameters
            ----------
            desired_pressure : float
                Desired pressure as input to the system
            steps : int, optional
                The number of steps to perform the simulation for
        """

        for iter in range(steps):
            self.step_mass_spring_damper(x_desired=desired_pressure)
            self.map_pressure_to_frequency(pressure=self.x)


def run_simulation_sequence(pacifier, dt_override=None):
    """Run the same simulation sequence on both implementations"""
    
    if dt_override is not None:
        pacifier.dt = dt_override
    
    # Run the exact same sequence as in the original files
    duration = 1.22
    pacifier.run(desired_pressure=0, steps=int(duration/pacifier.dt))

    duration = 1.22
    pacifier.run(desired_pressure=0.2, steps=int(duration/pacifier.dt))

    duration = 0.89
    pacifier.run(desired_pressure=0, steps=int(duration/pacifier.dt))

    duration = 0.97
    pacifier.run(desired_pressure=0.23, steps=int(duration/pacifier.dt))

    duration = 0.89
    pacifier.run(desired_pressure=0, steps=int(duration/pacifier.dt))

    duration = 1.05
    pacifier.run(desired_pressure=0.14, steps=int(duration/pacifier.dt))

    duration = 0.97
    pacifier.run(desired_pressure=0, steps=int(duration/pacifier.dt))

    duration = 1.14
    pacifier.run(desired_pressure=0.17, steps=int(duration/pacifier.dt))

    duration = 1.14
    pacifier.run(desired_pressure=0, steps=int(duration/pacifier.dt))

    duration = 1.30
    pacifier.run(desired_pressure=0.16, steps=int(duration/pacifier.dt))

    duration = 1.62
    pacifier.run(desired_pressure=0, steps=int(duration/pacifier.dt))

    duration = 1.95
    pacifier.run(desired_pressure=0.11, steps=int(duration/pacifier.dt))

    duration = 1.95
    pacifier.run(desired_pressure=0.5, steps=int(duration/pacifier.dt))

    duration = 2.25
    pacifier.run(desired_pressure=0, steps=int(duration/pacifier.dt))

    duration = 2.5
    pacifier.run(desired_pressure=0.9, steps=int(duration/pacifier.dt))


def compare_implementations(condition="analog", save_plots=False):
    """Compare the pressure trajectories of both implementations"""
    
    # Initialize both implementations
    print(f"Comparing implementations for condition: {condition}")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    pac_v4 = PacifierV4(condition=condition)
    
    np.random.seed(42)  # Reset seed for fair comparison
    pac_v6 = PacifierV6(condition=condition, dt=0.01)
    
    # For V4, we need to adjust dt to match the original usage
    if condition == "non-analog":
        dt_v4 = 0.1  # As used in the original v4 main
    else:
        dt_v4 = 0.01  # Use same as v6 for better comparison
    
    # Run simulations
    print("Running V4 simulation...")
    run_simulation_sequence(pac_v4, dt_override=dt_v4)
    
    print("Running V6 simulation...")
    run_simulation_sequence(pac_v6)
    
    # Create comparison plot - single plot with both trajectories
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot both implementations in one plot
    time_points_v4 = np.arange(len(pac_v4.x_log)) * pac_v4.dt
    time_points_v6 = np.arange(len(pac_v6.x_log)) * pac_v6.dt
    
    # Plot desired pressure only once (they should be the same)
    ax.plot(time_points_v6, pac_v6.x_desired_log, label='Desired', color="orange", linewidth=2, alpha=0.7)
    
    # V4 (old) implementation with dashed lines
    ax.plot(time_points_v4, pac_v4.x_log, label='Old Bio Mdl', color="blue", linewidth=2, linestyle='--', alpha=0.8)
    
    # V6 (current) implementation with solid lines
    ax.plot(time_points_v6, pac_v6.x_log, label='New Bio Mdl', color="blue", linewidth=2, alpha=0.8)
    
    # Threshold line
    ax.axhline(y=0.1, color='black', linestyle=':', alpha=0.7, label='Threshold')
    
    ax.set_ylabel('Pressure (psi)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Pressure Trajectory Comparison - {condition.title()} Condition')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/comparison_{condition}.pdf'
        plt.savefig(filename)
        print(f"Comparison plot saved to: {filename}")
        filename = f'/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/comparison_{condition}.png'
        plt.savefig(filename)
        print(f"Comparison plot saved to: {filename}")
    
    plt.show()
    
    # Print some statistics for comparison
    print(f"\n=== Comparison Statistics ({condition}) ===")
    print(f"V4 - Final time: {pac_v4.time:.3f}s, Final pressure: {pac_v4.x[0]:.3f}, Max pressure: {np.max(pac_v4.x_log):.3f}")
    print(f"V6 - Final time: {pac_v6.time:.3f}s, Final pressure: {pac_v6.x:.3f}, Max pressure: {np.max(pac_v6.x_log):.3f}")
    print(f"V4 - dt: {pac_v4.dt}, Number of steps: {len(pac_v4.x_log)}")
    print(f"V6 - dt: {pac_v6.dt}, Number of steps: {len(pac_v6.x_log)}")
    
    return pac_v4, pac_v6


def overlay_comparison(condition="analog", save_plots=True):
    """Create an overlay comparison of both implementations"""
    
    print(f"Creating overlay comparison for condition: {condition}")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    pac_v4 = PacifierV4(condition=condition)
    
    np.random.seed(42)  # Reset seed for fair comparison
    pac_v6 = PacifierV6(condition=condition, dt=0.01)
    
    # For V4, we need to adjust dt to match the original usage
    if condition == "non-analog":
        dt_v4 = 0.1  # As used in the original v4 main
    else:
        dt_v4 = 0.01  # Use same as v6 for better comparison
    
    # Run simulations
    run_simulation_sequence(pac_v4, dt_override=dt_v4)
    run_simulation_sequence(pac_v6)
    
    # Create overlay plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot both implementations
    time_points_v4 = np.arange(len(pac_v4.x_log)) * pac_v4.dt
    time_points_v6 = np.arange(len(pac_v6.x_log)) * pac_v6.dt
    
    # Plot desired pressure only once
    ax.plot(time_points_v6, pac_v6.x_desired_log, label='Desired', color="orange", linewidth=2, alpha=0.6, linestyle='--')
    
    # V4 (old) implementation with dashed lines
    ax.plot(time_points_v4, pac_v4.x_log, label='Old Bio Mdl', color="blue", linewidth=2, linestyle='--', alpha=0.8)
    
    # V6 (current) implementation with solid lines
    ax.plot(time_points_v6, pac_v6.x_log, label='New Bio Mdl', color="blue", linewidth=2, alpha=0.8)
    
    # Threshold line
    ax.axhline(y=0.1, color='black', linestyle=':', alpha=0.7, label='Threshold')
    
    ax.set_ylabel('Pressure (psi)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Pressure Trajectory Comparison - {condition.title()} Condition')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/overlay_comparison_{condition}.pdf'
        plt.savefig(filename)
        print(f"Overlay comparison plot saved to: {filename}")
    
    plt.show()
    
    return pac_v4, pac_v6


if __name__ == "__main__":
    print("=== Pressure2Sound Implementation Comparison ===")
    print("This script compares V4 (usedInFirstPaper) and V6 (current) implementations")
    print()
    
    # Compare analog condition
    print("1. Comparing ANALOG condition:")
    pac_v4_analog, pac_v6_analog = compare_implementations(condition="analog")
    print()
    
    # # Compare non-analog condition  
    # print("2. Comparing NON-ANALOG condition:")
    # pac_v4_nonanalog, pac_v6_nonanalog = compare_implementations(condition="non-analog")
    # print()
    
    # # Create overlay plots
    # print("3. Creating overlay comparisons:")
    # overlay_comparison(condition="analog")
    # overlay_comparison(condition="non-analog")
    
    print("\n=== Comparison Complete ===")
    print("Check the generated PDF files for visual comparison of the implementations.")
