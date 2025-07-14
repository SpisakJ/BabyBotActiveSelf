import numpy as np
import matplotlib.pyplot as plt
# import sounddevice as sd
from scipy.linalg import expm

class Pacifier:
    def __init__(self, condition="analog", dt=0.01, enable_adaptive_gain=False, enable_noise=False, enable_selective_integrator=False):
        
        self.condition = condition
        self.enable_adaptive_gain = enable_adaptive_gain
        self.enable_noise = enable_noise
        self.enable_selective_integrator = enable_selective_integrator
        
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

        # Check if any of the features requiring v4 parameters are enabled
        self.use_v4_system = enable_adaptive_gain or enable_noise or enable_selective_integrator
        
        if self.use_v4_system:
            # Use Pressure2Soundv4 parameters and system
            self.m = 1
            self.k = 15
            self.d = 1.5*np.sqrt(self.m*self.k)
            self.dt = dt if dt != 0.01 else 0.001  # Use 0.001 as default for v4
            
            # v4 specific variables for noise and adaptive features
            self.equ_noise = 0
            self.x_desired_noise = 0
            self.F_noise = 0
        else:
            # Use Pressure2Soundv6 parameters and system
            self.m = 0.2
            self.k = 15.0
            self.d = 1.7*np.sqrt(self.m*self.k)
            self.k_I = self.k*2.2
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
        
        if self.use_v4_system:
            # Use Pressure2Soundv4 approach with adaptive gain, noise, and selective integrator
            if self.enable_noise:
                if x_desired != self.x_desired_log[-1]:
                    self.x_desired_noise = np.random.normal(0, 0.01, 1)
                    self.equ_noise = np.random.normal(-0.01, 0.01, 1)
                
                if self.F_noise == 0:
                    self.F_noise = np.random.normal(0.006, 0.006/2, 1)
            else:
                self.x_desired_noise = 0
                self.equ_noise = 0
                self.F_noise = 0
            
            x_target = x_desired + self.x_desired_noise
            err = (x_target - self.x)
            self.err_int = self.err_int + err

            if self.enable_selective_integrator and x_target > 0.05:
                # Selective integrator active when pressure is above threshold
                if self.enable_adaptive_gain:
                    k_I = (np.tanh(80*abs(err))+1)*(0.008+self.F_noise) * (self.dt/0.001)
                    F_u = k_I*self.err_int
                else:
                    k_I = 0.03 * (self.dt/0.001) #0.015
                    F_u = k_I*self.err_int

                F = self.F + 1*(F_u-self.F)
            elif not self.enable_selective_integrator:
                # Standard integrator behavior
                if self.enable_adaptive_gain:
                    k_I = (np.tanh(80*abs(err))+1)*(0.008+self.F_noise) * (self.dt/0.001)
                    F_u = k_I*self.err_int
                else:
                    k_I = 0.03 * (self.dt/0.001)
                    F_u = k_I*self.err_int

                F = self.F + 1*(F_u-self.F)
            else:
                # Selective integrator: reset when pressure is low
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
            self.step = self.step + 1
            self.time = self.step*self.dt
            
        else:
            # Use Pressure2Soundv6 approach with exact ZOH discretization
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

    def visualize_system(self, show_pitch=False):
        """
            Plot pressure, force and frequencies.
        """
        # Initialize subplots
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 3))
        time_points = np.arange(len(self.x_log)) * self.dt
        
        # Plot pressure data on primary y-axis
        ax1.plot(time_points, self.x_log, label=r'$x$', color="blue")
        ax1.plot(time_points, self.x_desired_log, label=r'$x_{\rm des}$', color="orange")
        ax1.axhline(y=0.1, color='r', linestyle='--', label=r'$x_{\rm thr}$')
        ax1.set_ylabel('Pressure (psi)', color='black')
        ax1.set_ylim([-0.1, 1.1])
        ax1.grid(True)
        ax1.set_xlabel('Time (s)')
        
        if show_pitch:
            # Create secondary y-axis for pitch
            ax2 = ax1.twinx()
            ax2.plot(time_points, self.frequency_log, label=r'$p$', color="green")
            ax2.set_ylabel('Pitch', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax1.legend()
        
        plt.tight_layout()
        
        # Choose appropriate filename based on system type
        if self.use_v4_system:
            filename = '/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/BioMdl_result_v8.pdf'
        else:
            filename = '/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/BioMdl_result_v8.pdf'
        
        plt.savefig(filename)
        # plt.show()

if __name__ == "__main__":
    # Example usage with v6 system (no special features enabled)
    print("Testing v6 system (no special features)...")
    pac_env = Pacifier(condition="analog", dt=0.01, 
                          enable_adaptive_gain=False, 
                          enable_noise=False, 
                          enable_selective_integrator=False)

    duration = 1.22
    pac_env.run(desired_pressure=0, steps=int(duration/pac_env.dt))

    duration = 1.22
    pac_env.run(desired_pressure=0.2, steps=int(duration/pac_env.dt))

    duration = 0.89
    pac_env.run(desired_pressure=0, steps=int(duration/pac_env.dt))

    duration = 0.97
    pac_env.run(desired_pressure=0.23, steps=int(duration/pac_env.dt))

    duration = 0.89
    pac_env.run(desired_pressure=0, steps=int(duration/pac_env.dt))

    duration = 1.05
    pac_env.run(desired_pressure=0.14, steps=int(duration/pac_env.dt))

    duration = 0.97
    pac_env.run(desired_pressure=0, steps=int(duration/pac_env.dt))

    duration = 1.14
    pac_env.run(desired_pressure=0.17, steps=int(duration/pac_env.dt))

    duration = 1.14
    pac_env.run(desired_pressure=0, steps=int(duration/pac_env.dt))

    duration = 1.30
    pac_env.run(desired_pressure=0.16, steps=int(duration/pac_env.dt))

    duration = 1.62
    pac_env.run(desired_pressure=0, steps=int(duration/pac_env.dt))

    duration = 1.95
    pac_env.run(desired_pressure=0.11, steps=int(duration/pac_env.dt))

    duration = 1.95
    pac_env.run(desired_pressure=0.5, steps=int(duration/pac_env.dt))

    duration = 2.25
    pac_env.run(desired_pressure=0, steps=int(duration/pac_env.dt))

    duration = 2.5
    pac_env.run(desired_pressure=0.9, steps=int(duration/pac_env.dt))

    pac_env.visualize_system(show_pitch=True)

    print("end")