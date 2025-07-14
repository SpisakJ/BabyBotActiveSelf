import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pandas as pd
import os


def load_data_from_file(filepath):
    """Load pressure sequence data from recorded log file
    
    Parameters
    ----------
    filepath : str
        Path to the data file containing recorded pressure sequence
        
    Returns
    -------
    dict
        Dictionary containing loaded data arrays: f_log, x_log, x_desired, frequency, time
    """
    try:
        # Load the CSV data
        data = pd.read_csv(filepath)
        
        # Extract the columns
        loaded_data = {
            'f_log': data['f_log'].values,
            'x_log': data['x_log'].values, 
            'x_desired': data['x_desired'].values,
            'frequency': data['frequency'].values,
            'time': data['time'].values
        }
        
        print(f"Loaded data from {filepath}")
        print(f"  Duration: {loaded_data['time'][-1]:.2f}s")
        print(f"  Data points: {len(loaded_data['time'])}")
        print(f"  Max desired pressure: {np.max(loaded_data['x_desired']):.3f}")
        print(f"  Min desired pressure: {np.min(loaded_data['x_desired']):.3f}")
        
        return loaded_data
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None


def run_simulation_from_data(pacifier, data_file_path, dt_override=None):
    """Run simulation using desired pressure sequence from a data file
    
    Parameters
    ----------
    pacifier : PacifierV4 or PacifierV6
        The pacifier instance to run simulation on
    data_file_path : str
        Path to the data file containing the desired pressure sequence
    dt_override : float, optional
        Override the default dt of the pacifier
    """
    
    if dt_override is not None:
        pacifier.dt = dt_override
    
    # Load the data
    loaded_data = load_data_from_file(data_file_path)
    if loaded_data is None:
        return False
    
    # Get the desired pressure sequence and time points
    x_desired_sequence = loaded_data['x_desired']
    time_sequence = loaded_data['time']
    
    # Run simulation step by step following the loaded sequence
    for i, desired_pressure in enumerate(x_desired_sequence):
        pacifier.run(desired_pressure=desired_pressure, steps=1)
    
    print(f"Simulation completed using data from {data_file_path}")
    return True


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
        self.m  = 0.2
        self.k  = 15.0
        self.d  = 1.7*np.sqrt(self.m*self.k)
        self.k_I= self.k*0.9 #1.7 #*2.2
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


def compare_implementations(condition="analog", save_plots=True):
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
    ax.axhline(y=0.2, color='black', linestyle=':', alpha=0.7, label='Threshold')
    
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
    print(f"V4 - Final time: {pac_v4.time:.3f}s, Final pressure: {pac_v4.x:.3f}, Max pressure: {np.max(pac_v4.x_log):.3f}")
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
    ax.axhline(y=0.2, color='black', linestyle=':', alpha=0.7, label='Threshold')
    
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


def compare_implementations_with_data(data_file_path, condition="analog", save_plots=True):
    """Compare implementations using desired pressure sequence from a data file"""
    
    # Check if data file exists
    if not os.path.exists(data_file_path):
        print(f"Error: Data file {data_file_path} not found!")
        return None, None
    
    # Initialize both implementations
    print(f"Comparing implementations for condition: {condition}")
    print(f"Using data from: {data_file_path}")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    pac_v4 = PacifierV4(condition=condition)
    
    np.random.seed(42)  # Reset seed for fair comparison
    pac_v6 = PacifierV6(condition=condition, dt=0.01)
    
    # For V4, we need to adjust dt to match the original usage
    if condition == "non-analog":
        dt_v4 = 0.01 #0.1  # As used in the original v4 main
    else:
        dt_v4 = 0.01  # Use same as v6 for better comparison
    
    # Run simulations using data from file
    print("Running V4 simulation with loaded data...")
    success_v4 = run_simulation_from_data(pac_v4, data_file_path, dt_override=dt_v4)
    
    print("Running V6 simulation with loaded data...")
    success_v6 = run_simulation_from_data(pac_v6, data_file_path)
    
    if not (success_v4 and success_v6):
        print("Error: Failed to run simulations with loaded data")
        return None, None
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Pressure comparison
    time_points_v4 = np.arange(len(pac_v4.x_log)) * pac_v4.dt
    time_points_v6 = np.arange(len(pac_v6.x_log)) * pac_v6.dt
    
    # Plot desired pressure
    ax1.plot(time_points_v6, pac_v6.x_desired_log, label='Desired', color="orange", linewidth=2, alpha=0.7)
    
    # V4 (old) implementation with dashed lines
    ax1.plot(time_points_v4, pac_v4.x_log, label='Old Bio Mdl', color="blue", linewidth=2, linestyle='--', alpha=0.8)
    
    # V6 (current) implementation with solid lines
    ax1.plot(time_points_v6, pac_v6.x_log, label='New Bio Mdl', color="blue", linewidth=2, alpha=0.8)
    
    # Threshold line
    ax1.axhline(y=0.2, color='black', linestyle=':', alpha=0.7, label='Threshold')
    
    ax1.set_ylabel('Pressure (psi)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title(f'Pressure Trajectory Comparison - {condition.title()} Condition (Data from {os.path.basename(data_file_path)})')
    ax1.set_ylim([-0.1, max(1.1, np.max(pac_v6.x_desired_log) * 1.1)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency comparison
    ax2.plot(time_points_v4, pac_v4.frequency_log, label='Old Bio Mdl Frequency', color="red", linewidth=2, linestyle='--', alpha=0.8)
    ax2.plot(time_points_v6, pac_v6.frequency_log, label='New Bio Mdl Frequency', color="red", linewidth=2, alpha=0.8)
    
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title(f'Frequency Trajectory Comparison - {condition.title()} Condition')
    ax2.set_ylim([0, 420])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        data_name = os.path.splitext(os.path.basename(data_file_path))[0]
        filename = f'/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/comparison_{condition}_{data_name}.pdf'
        plt.savefig(filename)
        print(f"Comparison plot saved to: {filename}")
        filename = f'/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/comparison_{condition}_{data_name}.png'
        plt.savefig(filename)
        print(f"Comparison plot saved to: {filename}")
    
    plt.show()
    
    # Print some statistics for comparison
    print(f"\n=== Comparison Statistics ({condition}, {os.path.basename(data_file_path)}) ===")
    print(f"V4 - Final time: {pac_v4.time:.3f}s, Final pressure: {pac_v4.x[0]:.3f}, Max pressure: {np.max(pac_v4.x_log):.3f}")
    print(f"V6 - Final time: {pac_v6.time:.3f}s, Final pressure: {pac_v6.x:.3f}, Max pressure: {np.max(pac_v6.x_log):.3f}")
    print(f"V4 - dt: {pac_v4.dt}, Number of steps: {len(pac_v4.x_log)}")
    print(f"V6 - dt: {pac_v6.dt}, Number of steps: {len(pac_v6.x_log)}")
    
    # Calculate some performance metrics
    desired_pressure = pac_v6.x_desired_log
    
    # Root Mean Square Error
    rmse_v4 = np.sqrt(np.mean((pac_v4.x_log[:len(desired_pressure)] - desired_pressure)**2))
    rmse_v6 = np.sqrt(np.mean((pac_v6.x_log[:len(desired_pressure)] - desired_pressure)**2))
    
    print(f"RMSE - V4: {rmse_v4:.6f}, V6: {rmse_v6:.6f}")
    
    # Mean Absolute Error
    mae_v4 = np.mean(np.abs(pac_v4.x_log[:len(desired_pressure)] - desired_pressure))
    mae_v6 = np.mean(np.abs(pac_v6.x_log[:len(desired_pressure)] - desired_pressure))
    
    print(f"MAE  - V4: {mae_v4:.6f}, V6: {mae_v6:.6f}")
    
    return pac_v4, pac_v6


def compare_multiple_data_files(data_folder, condition="analog", max_files=5):
    """Compare implementations using multiple data files from the data folder"""
    
    data_files = []
    for filename in os.listdir(data_folder):
        if filename.startswith("johannes_data_"):
            data_files.append(os.path.join(data_folder, filename))
    
    data_files.sort()  # Sort for consistent ordering
    data_files = data_files[:max_files]  # Limit number of files
    
    print(f"Found {len(data_files)} data files, processing first {len(data_files)}:")
    
    results = []
    for data_file in data_files:
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(data_file)}")
        print('='*50)
        
        pac_v4, pac_v6 = compare_implementations_with_data(data_file, condition=condition, save_plots=True)
        if pac_v4 is not None and pac_v6 is not None:
            results.append((os.path.basename(data_file), pac_v4, pac_v6))
    
    return results


if __name__ == "__main__":
    print("=== Pressure2Sound Implementation Comparison ===")
    print("This script compares V4 (usedInFirstPaper) and V6 (current) implementations")
    print()
    
    # Options for different comparison modes
    use_data_files = True  # Set to False to use manual sequences
    all_data_files = False  # Set to True to iterate over all available log files
    
    if use_data_files:
        print("Using recorded data files for comparison...")
        
        # Define data folder path
        data_folder = "/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/data"
        
        if all_data_files:
            print("\nProcessing ALL available data files:")
            # Process all data files
            results_analog = compare_multiple_data_files(data_folder, condition="analog", max_files=100)
            
            print(f"\nProcessed {len(results_analog)} data files successfully.")
            
        else:
            # Compare with a specific data file
            print("\n1. Comparing with specific data file (johannes_data_0):")
            data_file_path = os.path.join(data_folder, "johannes_data_0")
            
            print("\nAnalog condition with data file:")
            pac_v4_data_analog, pac_v6_data_analog = compare_implementations_with_data(
                data_file_path, condition="analog", save_plots=True)
        
    else:
        print("Using manual pressure sequences for comparison...")
        
        # Compare analog condition
        print("1. Comparing ANALOG condition:")
        pac_v4_analog, pac_v6_analog = compare_implementations(condition="analog")
        print()
    
    print("\n=== Comparison Complete ===")
    print("Check the generated PDF files for visual comparison of the implementations.")
