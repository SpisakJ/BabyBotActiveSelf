import numpy as np
import matplotlib.pyplot as plt
# import sounddevice as sd
from scipy.linalg import expm

class Pacifier:
    def __init__(self, condition="analog", dt=0.01):
        
        self.condition = condition
        
        # for analog condition
        self.pressure_range = (0, 1)
        self.frequency_range = (0, 400)
        self.pressure_threshold = 0.16  # Pressure threshold for producing sound

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
        self.k_I= self.k*2.2
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

    def visualize_system(self, show_pitch=False):
        """
            Plot pressure, force and frequencies.
        """

        # Initialize subplots
        fig, axs = plt.subplots(1, 1, figsize=(5, 3))  # Corrected to plt.subplots

        time_points = np.arange(len(self.x_log)) * self.dt

        axs.plot(time_points, self.x_log, label=r'$x$', color="blue")
        axs.plot(time_points, self.x_desired_log, label=r'$x_{\rm des}$', color="orange")
        if show_pitch:
            axs.plot(time_points, self.frequency_log, label=r'$p$', color="green")
        axs.axhline(y=0.1, color='r', linestyle='--', label=r'$x_{\rm thr}$')
        axs.set_ylabel('Pressure (psi)')
        # axs.set_ylim([-0.1, 1.1])
        # axs.set_ylim([-0.1, 0.7])
        axs.legend()
        axs.grid(True)
        # axs.set_xlim((time_points[0], time_points[-1]))
        # axs.set_xlim((0, 13.5))
        axs.set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig('/home/jheidersberger/Documents/Projects/BabyBotActiveSelf/Pressure2Sound_JH/BioMdl_result_2.pdf')
        plt.show()

if __name__ == "__main__":
    pac_env = Pacifier(condition="analog", dt=0.01)

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

    duration = 5.5
    pac_env.run(desired_pressure=1.0, steps=int(duration/pac_env.dt))

    duration = 7.5
    pac_env.run(desired_pressure=0.0, steps=int(duration/pac_env.dt))

    pac_env.visualize_system()

    # pac_env.play_dynamic_pitch()

    print("end")
