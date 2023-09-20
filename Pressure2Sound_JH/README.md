# Pressure2Sound Functions

The `Pressure2Sound.py` file contains several functions to simulate and map pressure changes to sound frequencies. Below is an overview of these functions:

## Table of Contents
1. [simulate_mass_spring_damper](#simulate_mass_spring_damper)
2. [map_pressure_to_frequency](#map_pressure_to_frequency)
    - [map_pressure_to_frequency_proportional](#map_pressure_to_frequency_proportional)
    - [map_pressure_to_frequency_rand](#map_pressure_to_frequency_rand)
3. [play_dynamic_pitch](#play_dynamic_pitch)

---

### 1. simulate_mass_spring_damper

- **Usage**: Simulates the pressure change over time based on a desired pressure time series.
- **Parameters**: 
    - `pressure_desired`: Desired pressure time series
    - `time`: Time vector
    - `noiseFactor`: Factor for noise simulation
- **Returns**: Simulated pressure time series

---

### 2. map_pressure_to_frequency

- **Usage**: Maps a pressure time series to a frequency time series based on the given condition.
- **Parameters**: 
    - `pressure`: Pressure time series
    - `time`: Time vector
    - `condition`: 'analog' or 'non-analog'
- **Returns**: Frequency time series

#### 2.1 map_pressure_to_frequency_proportional

- **Sub-function of**: `map_pressure_to_frequency`
- **Usage**: Maps pressure to frequency for analog conditions.
- **Parameters**: 
    - `pressure`: Pressure value
- **Returns**: Mapped frequency value

#### 2.2 map_pressure_to_frequency_rand

- **Sub-function of**: `map_pressure_to_frequency`
- **Usage**: Maps pressure to frequency for non-analog conditions, i.e. trill.
- **Parameters**: 
    - `pressure`: Pressure value
    - `time`: Time value
- **Returns**: Mapped frequency value (trill)

---

### 3. play_dynamic_pitch

- **Usage**: Plays sounds based on a given time series of frequency.
- **Parameters**: 
    - `frequency`: Frequency time series
    - `time`: Time vector
- **Returns**: None (plays sound)

