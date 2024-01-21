# Pressure2Sound Functions

The `Pressure2Soundv3.py` file contains several functions to simulate the pressure and map the pressure to sound frequencies. Below is an overview of these functions:

## Table of Contents
1. [step_mass_spring_damper](#step_mass_spring_damper)
2. [map_pressure_to_frequency](#map_pressure_to_frequency)
    - [map_pressure_to_frequency_proportional](#map_pressure_to_frequency_proportional)
    - [map_pressure_to_frequency_rand](#map_pressure_to_frequency_rand)
3. [run](#run)
4. [visualize_system](#visualize_system)
5. [play_dynamic_pitch](#play_dynamic_pitch)

---

### 1. step_mass_spring_damper

- **Usage**: Simulates the pressure change based on a desired pressure for one timestep.
- **Parameters**: 
    - `pressure_desired`: Desired pressure value
- **Returns**: None

---

### 2. map_pressure_to_frequency

- **Usage**: Maps a pressure to a frequency based on the given condition.
- **Parameters**: 
    - `pressure`: Pressure time series
- **Returns**: None

#### 2.1 map_pressure_to_frequency_proportional

- **Sub-function of**: `map_pressure_to_frequency`
- **Usage**: Maps pressure to frequency for analog conditions.
- **Parameters**: 
    - `pressure`: Pressure value
- **Returns**: None

#### 2.2 map_pressure_to_frequency_rand

- **Sub-function of**: `map_pressure_to_frequency`
- **Usage**: Maps pressure to frequency for non-analog conditions, i.e. trill.
- **Parameters**: 
    - `pressure`: Pressure value
- **Returns**: None

---

### 3. run

- **Usage**: Simulates the system for a specified number of steps.
- **Parameters**: 
    - `desired_pressure`: Desired pressure as system input
    - `steps`: Number of steps for which to perform the simulation
- **Returns**: None

---

### 4. visualize_system

- **Usage**: Plots the pressure, force and frequency.
- **Parameters**: None
- **Returns**: None

---

### 5. play_dynamic_pitch

- **Usage**: Plays sounds for the computed frequencies.
- **Parameters**: None
- **Returns**: None (plays sound)

