#%%
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np
import matplotlib
def setupPlot():
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time Step')
    plt.ylabel('Pressure Value')
    plt.grid(True)
def clean_x(val):
    # If val is a string like "[0.123]", convert to float
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):

        try:
            return float(val.strip("[]"))
        except Exception:
            return np.nan
    return val
def plotAblation(key, colors=['blue', 'orange']):
    anadata = []
    nonanadata = []
    for filename in glob.glob("./Ablations_06_08_2025/"+key+"/pressures_age_old_start_analog_*.csv"):
        df = pd.read_csv(filename)
        # Clean and overwrite the x column
        df["x"] = [clean_x(v) for v in df["x"].values]
        # Save the cleaned column back to the file
        df.to_csv(filename, index=False)
        x_vals = df["x"].values
        anadata.append(x_vals)
        print(f"Loaded {filename} with {len(x_vals)} entries.")
    for filename in glob.glob("./Ablations_06_08_2025/"+key+"/pressures_age_old_start_non-analog_*.csv"):
        df = pd.read_csv(filename)
        df["x"] = [clean_x(v) for v in df["x"].values]
        df.to_csv(filename, index=False)
        x_vals = df["x"].values
        nonanadata.append(x_vals)
        print(f"Loaded {filename} with {len(x_vals)} entries.")
    data_array_ana = np.array(anadata)
    data_array_nonana = np.array(nonanadata)


    mean_data_ana = np.mean(data_array_ana, axis=0)
    mean_data_nonana = np.mean(data_array_nonana, axis=0)

    std_data_ana = np.std(data_array_ana, axis=0)
    std_data_nonana = np.std(data_array_nonana, axis=0)

    plt.plot(mean_data_ana, label=f'Mean Analog Start {key}', color=colors[0])
    plt.plot(mean_data_nonana, label=f'Mean Non-Analog Start {key}', color=colors[1])

    plt.fill_between(range(len(mean_data_ana)), 
                    mean_data_ana - std_data_ana, 
                    mean_data_ana + std_data_ana, 
                    alpha=0.2, color=colors[0])
    plt.fill_between(range(len(mean_data_nonana)), 
                    mean_data_nonana - std_data_nonana, 
                    mean_data_nonana + std_data_nonana, 
                    alpha=0.2, color=colors[1])
    plt.legend()
    
# Example usage:
setupPlot()
#plotAblation("longer", colors=['blue', 'orange'])
#plotAblation("noExh2", colors=['blue', 'orange'])
#plotAblation("noExpnoExh", colors=['blue', 'orange'])
#plotAblation("onlyExhconstCond", colors=['purple', 'yellow'])
#plotAblation("onlyExhconstCondWithInst", colors=['violet', 'red'])
plotAblation("allLossesNormPastConstInst", colors=['green', 'lightgreen'])
plotAblation("noExplorationNormPast", colors=['cyan', 'lightblue'])
#plotAblation("mechanical_model_all_true", colors=['red', 'green'])
#plotAblation("mechanical_model_all_true_dt_01", colors=['purple', 'brown'])
#plotAblation("mechanical_model_gain_false", colors=['purple', 'brown'])
#plotAblation("mechanical_model_gain_false_dt_01", colors=['cyan', 'magenta'])
#plotAblation("mechanical_model_noise_false", colors=['cyan', 'magenta'])
#plotAblation("mechanical_model_noise_false_dt_01", colors=['lightblue', 'lightgreen'])
#plotAblation("mechanical_model_integrator_false", colors=['yellow', 'gray'])
#plotAblation("mechanical_model_integrator_false_dt_01", colors=['yellow', 'gray'])
#plotAblation("mechanical_model_gain_true", colors=['black', 'pink'])
#plotAblation("mechanical_model_gain_true_dt_01", colors=['skyblue', 'plum'])
#plotAblation("mechanical_model_noise_true", colors=['teal', 'coral'])
#plotAblation("mechanical_model_noise_true_dt_01", colors=['navy', 'gold'])
#plotAblation("mechanical_model_integrator_true", colors=['lime', 'salmon'])
#plotAblation("mechanical_model_integrator_true_dt_01", colors=['olive', 'violet'])
plt.title('Comparison of Different Ablations')
plt.show()
plt.savefig("noExhDesiredPressureIsOutputPressure.png")
