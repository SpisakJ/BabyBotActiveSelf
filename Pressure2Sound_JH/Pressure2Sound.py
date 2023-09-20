# %%
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


class Pacifier:
    def __init__(self):
        self.voltage_range = (0.5, 4.5)
        self.pressure_range = (0, 1)
        # self.frequency_range = (0, 400)
        self.frequency_range = (0, 1000)
        self.db = 65
        self.duration = 0.2
        self.pressure_threshold = 0.1  # Pressure threshold for producing sound

        # for non-analog condition
        self.trill_t0 = 0.0
        self.trill_freq = 0.0
        self.trill_duration = 0.55 #0.55
        self.n_trills = 11 #11
        self.trill_freq_range = (0, 400) #(0, 400)
    
    def map_pressure_to_frequency_proportional(self, pressure):
        if pressure > self.pressure_threshold:
            pressure_min, pressure_max = self.pressure_range
            frequency_min, frequency_max = self.frequency_range
            pressure = max(min(pressure, pressure_max), pressure_min)  # Clamp pressure within range
            frequency = ((pressure - pressure_min) / (pressure_max - pressure_min)) * (frequency_max - frequency_min) + frequency_min
        else:
            frequency = 0.0
        # print("frequency", frequency)
        return frequency
    
    def map_pressure_to_frequency_rand(self, pressure, time):
        time_diff = time - self.trill_t0
        frequency = 0.0
        if time_diff > self.trill_duration or self.trill_t0 == 0.0:
            if pressure > self.pressure_threshold:
                self.trill_t0 = time
                self.trill_freq = np.random.uniform(self.trill_freq_range[0], self.trill_freq_range[1], self.n_trills)
                frequency = self.trill_freq[-1]
        else:
            for i in range(0, self.n_trills, 1):                
                lower_bound = self.trill_duration - (i + 1) * (self.trill_duration/self.n_trills)
                upper_bound = self.trill_duration - i * (self.trill_duration/self.n_trills)
                if lower_bound < time_diff <= upper_bound:
                    frequency = self.trill_freq[i]
                    break
            
        return frequency
    
    def play_dynamic_pitch(self, pitch_series, pitch_times, duration):
        sample_rate = 44100
        time = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Interpolate the pitch values to match the audio sample rate
        interpolated_pitch = np.interp(time, pitch_times, pitch_series)
        
        # Generate an FM-modulated waveform using the interpolated pitch values
        modulation_frequency = interpolated_pitch
        phase = np.cumsum(2 * np.pi * modulation_frequency / sample_rate)
        waveform = np.sin(phase)
        
        sd.play(waveform, sample_rate)
        sd.wait()
    
    def map_pressure_to_frequency(self, pressures, times, condition='analog'):
        frq_log = []
        
        if condition == "analog":
            for pressure in pressures:
                frequency = pac.map_pressure_to_frequency_proportional(pressure)
                frq_log.append(frequency)
        elif condition == "non-analog":
            for i in range(pressures.shape[0]):
                frequency = pac.map_pressure_to_frequency_rand(pressures[i], times[i])
                frq_log.append(frequency)
        
        return np.array(frq_log)

    def simulate_mass_spring_damper(self, x_desired, time, k_noise=0.0):
        m = 0.1
        k = 2.0
        c = 2*np.sqrt(m*k)*0.75
        F_constant = k*1.5#0.80
        # F_constant = k*6
        F_old = 0
        x_initial = 0.0
        v_initial = 0.0

        position = np.zeros(time.shape)
        velocity = np.zeros(time.shape)
        force = np.zeros(time.shape)

        x_desired_prev = 0
        
        x = x_initial
        v = v_initial
        reached_desired_position = False  # Flag to track if desired position has been reached
        
        for i in range(time.shape[0]):
            position[i] = x
            velocity[i] = v
            
            if x_desired[i] != x_desired_prev:  # Check if desired position has changed
                reached_desired_position = False  # Reset the flag if desired position changes
                x_desired_prev = x_desired[i]
            
            if not reached_desired_position:
                noise = np.random.normal(0,1,1)
                F_u = F_constant * np.sign(x_desired[i] - x)
                # F_u = F_constant * np.sign(x_desired[i] - x) * np.clip(np.linalg.norm(x_desired[i] - x), 0.1, 1)
                F = F_old + 0.001*(F_u-F_old)
                # F = F_old + 0.001*(F_u-F_old)
                if np.isclose(x, x_desired[i], atol=0.025):  # Check if desired position is reached
                    reached_desired_position = True
                    F = 0.0  # Set force to zero once desired position is reached
            else:
                F = 0.0  # Force is zero after desired position is reached
            
            F = F  + noise*k_noise*F_constant*F

            F_old = F

            force[i] = F
            
            a = (F - c * v - k * x) / m
            v = v + a * dt
            x = x + v * dt
            
        return time, position, force
    
    def align_yaxis(self, ax1, ax2):
        """Align zeros of the two axes, zooming them out by same ratio"""
        axes = np.array([ax1, ax2])
        extrema = np.array([ax.get_ylim() for ax in axes])
        tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
        # Ensure that plots (intervals) are ordered bottom to top:
        if tops[0] > tops[1]:
            axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

        # How much would the plot overflow if we kept current zoom levels?
        tot_span = tops[1] + 1 - tops[0]

        extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
        extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
        [axes[i].set_ylim(*extrema[i]) for i in range(2)]

    def bin_signal(self, signal, step_size=0.025, signal_range=[0.0, 0.6]):
        if len(signal_range) == 2:
            clipped_signal = np.clip(signal, signal_range[0], signal_range[1])
        else:
            clipped_signal = signal
        # Bin the clipped signal
        binned_signal = np.ceil(clipped_signal / step_size) * step_size
        return binned_signal

if __name__ == "__main__":
    pac = Pacifier()

    duration = 15
    dt = 0.001
    times = np.arange(0, duration, dt)
    x_desired = np.ones(times.shape)
    x_desired[:] = 0.5
    x_desired[int(2/dt):] = 0.4
    x_desired[int(4/dt):] = 0.6
    x_desired[int(6.5/dt):] = 0.2
    x_desired[int(7.5/dt):] = 0.3
    x_desired[int(11/dt):] = 0.1
    x_desired[int(12/dt):] = 0.0
    x_desired[int(13/dt):] = 0.1

    # Simulate the system
    times_nn, pressure_nn, force_nn = pac.simulate_mass_spring_damper(x_desired, times, k_noise=0.0)
    times, pressure, force = pac.simulate_mass_spring_damper(x_desired, times,k_noise=0.01)
    # times, pressure, force = pac.simulate_mass_spring_damper(x_desired, times,k_noise=0.0)
    frequencyArray = pac.map_pressure_to_frequency(pressure, times, condition="analog")

    # plt.figure(figsize=(10, 6))
    # plt.plot(times, pressure, label='Pressure')
    # plt.plot(times, x_desired, label='Desired pressure')
    # plt.axhline(y = 0.1, color = 'r', linestyle = '--')
    # plt.ylim((-0.1, 0.75))
    # plt.xlabel('Time')
    # plt.ylabel('Pressure [psi]')
    # plt.title('2nd order dynamical system')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Create the main figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot pressure data on the primary y-axis
    ax1.plot(times, x_desired, label='Desired pressure', color='orange')
    ax1.plot(times, pressure, label='Pressure', color='b')
    ax1.plot(times_nn, pressure_nn, label='Pressure without noise', color='m', linestyle='--')
    ax1.axhline(y = 0.1, color = 'r', linestyle = '--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Pressure [psi]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Create a secondary y-axis for the frequency data
    ax2 = ax1.twinx()
    ax2.plot(times, frequencyArray, label='Frequency', color='g')
    ax2.set_ylabel('Frequency [Hz]', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc="upper right")

    pac.align_yaxis(ax1, ax2)

    plt.show()

    fig2 = plt.figure(figsize=(10, 6))
    pressure_bin = pac.bin_signal(pressure)
    plt.plot(times, pressure, label='Pressure', color='b')
    plt.plot(times, pressure_bin, label='Pressure binned', color='g')
    plt.xlabel('Time')
    plt.ylabel('Pressure [psi]')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.yticks(np.arange(0, 0.6+0.025, 0.025))
    plt.show()

    pac.play_dynamic_pitch(frequencyArray, times, duration)