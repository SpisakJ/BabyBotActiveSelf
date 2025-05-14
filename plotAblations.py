#%%
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np

def setupPlot():
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time Step')
    plt.ylabel('Pressure Value')
    plt.grid(True)

def plotAblation(key, colors=['blue', 'orange']):
    anadata = []
    nonanadata = []
    for filename in glob.glob("./Ablations/"+key+"/pressures_age_old_start_analog_*.csv"):
        anadata.append(pd.read_csv(filename)["x"].values)
    for filename in glob.glob("./Ablations/"+key+"/pressures_age_old_start_non-analog_*.csv"):
        nonanadata.append(pd.read_csv(filename)["x"].values)

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
#plotAblation("prediction_5", colors=['blue', 'orange'])
plotAblation("standard", colors=['red', 'green'])
plt.title('Comparison of Different Ablations')
plt.show()