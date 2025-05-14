#%%
import pandas as pd
import numpy as np
from ast import literal_eval
import os
import glob
import matplotlib.pyplot as plt
#%%
all_files = glob.glob("./gridResCSVNew_31_03_25_*.csv")
li = []
for filename in all_files:
    print(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    print(df)

df = pd.concat(li, axis=0, ignore_index=True)
print(df)
#%%
df = pd.read_csv("gridResCSVNew_31_03_25.csv",index_col=0)#Combined.csv

#df2 = df[["exp","pred","exh","mem","stre","actu","sens","NAt","At"]]#.apply(literal_eval)#["exp","pred","exh","mem","stre","actu","sens","NAt","At"]]
#mat = df2.corr()
df2 = df.where(df["stre"]==0.45)
#df2 = df2.where(df2["exh"]==1)
#df2 = df2.where(df2["exp"]==0)
#df2 = df2.where(df2["pred"]==1)
#df2 = df2.where(df2["actu"]==0.05)
#df2 = df2.where(df2["sens"]==0.05)

df3 = df2
df3 = df3.dropna(how="any")
df3.corr()
#print(df2)
#%%
#df3.to_csv("gridResCSVNew_31_03_25.csv")
#%%
#df2 = df3
def expand_list_column(df, col_name, new_col_prefix):
    # Convert string representation of list to actual lists
    df[col_name] = df[col_name].apply(literal_eval)
    # Create new columns for each element of the list
    df[new_col_prefix + '_A'], df[new_col_prefix + '_N'] = zip(*df[col_name])
    return df.drop(columns=[col_name])  # Optionally drop the original column

df3 = expand_list_column(df3, 'NAt', 'NAt')
df3 = expand_list_column(df3, 'At', 'At')
df3.corr()
#%%
#df2 = df3
df2 = pd.read_csv("combined_median.csv")
#df2 = df2.where(df2["stre"]==0.25)
df2 = df2.where(df2["exh"]==0)
#df2 = df2.where(df2["exp"]==1)
#df2 = df2.where(df2["pred"]==1)
#df2 = df2.where(df2["actu"]==0.05)
#df2 = df2.where(df2["sens"]==0.05)
df2["NAt"] = df2["NAt_A"] +df2["At_A"]
df2 = df2.drop(columns=["At_A"],axis=1)
df2 = df2.drop(columns=["NAt_A"],axis=1)

df2["NNt"] = df2["NAt_N"] +df2["At_N"]
df2 = df2.drop(columns=["At_N"],axis=1)
df2 = df2.drop(columns=["NAt_N"],axis=1)

diff = df2["NAt"]-df2["NNt"]

more = (df2["NAt"]>df2["NNt"])/df2["NAt"].size


df2.insert(9,["A-N"],diff.values,True)
df2.insert(10,["more"],more,True)
mat = df2.corr(min_periods=3)
df2.corr(min_periods=3)
#plt.matshow(mat[9:])
#%%
ana = df2.where(df["exp"]==0)["NAt"]
nana = df2.where(df["exp"]==0)["NNt"]
more = (ana>nana).sum() /ana.size
print(more)
print(mat[:9])
'''
      NAt       NNt       A-N       moreA  
exp   0.187061 -0.065026  0.197894  0.210321  
pred -0.011532 -0.078432 -0.000920  0.010429  
exh   0.401600 -0.022377  0.408819  0.427596  
mem  -0.126481  0.079230 -0.138631 -0.175984  
stre -0.209005  0.573175 -0.289593 -0.319828  
actu -0.303691 -0.269200 -0.270002 -0.004931  
sens  0.018788 -0.061502  0.027397  0.018965  
'''
#%%

avgA = [0,0]
avgN = [0,0]
count = 0
exp = [0]
pred = [0,1,2]
exh = [2]
mem = [1,5]
stre = [0.65]
actu = [0.2]
sens = [0.2]
avg=[0,0]
#for rowA,rowN in zip(df.loc[(df["exp"].isin(exp)) & (df["pred"].isin(pred)) & (df["exh"].isin(exh)) & (df["mem"].isin(mem)) & (df["stre"].isin(stre)) & (df["actu"].isin(actu)) & (df["sens"].isin(sens))]["NAt"] ,
#                     df.loc[(df["exp"].isin(exp)) & (df["pred"].isin(pred)) & (df["exh"].isin(exh)) & (df["mem"].isin(mem)) & (df["stre"].isin(stre)) & (df["actu"].isin(actu)) & (df["sens"].isin(sens))]["AAt"]):
#    count+=1
#    avgA[0] += rowA[0]
#    avgA[1] += rowA[1]
#    avgN[0] += rowN[0]
#    avgN[1] += rowN[1]
#    avg[0] += rowN[0] +rowA[0]
#    avg[1] += rowN[1]+rowA[1]
for row in df.loc[(df["exp"].isin(exp))
                   & (df["pred"].isin(pred))
                     & (df["exh"].isin(exh))
                       & (df["mem"].isin(mem))
                         & (df["stre"].isin(stre))
                           & (df["actu"].isin(actu))
                            & (df["sens"].isin(sens))]["AAt"]:
    count+=1
    avg[0] += row[0]
    avg[1] += row[1]
print(count)
print([avgA[0]/count,avgA[1]/count],[avgN[0]/count,avgN[1]/count])
print([avg[0]/count*2,avg[1]/count*2])
if count != 486 and count != 729 and count != 1458 :
    print("WARNING some typo")

#[414.1639231824417, 277.14334705075447] [678.0946502057614, 135.08641975308643] this is base line avg over all
#[531.7201646090535, 272.24074074074076] [774.7448559670781, 128.85802469135803] if exp is 2
#[406.98148148148147, 270.2098765432099] [673.858024691358, 133.119341563786] if pred is 2
#[527.3230452674898, 276.76748971193416] [894.7674897119341, 135.01440329218107]if exh is 2
#[367.076817558299, 278.758573388203] [583.1495198902606, 145.519890260631] of mem is 5
#[226.68312757201647, 330.1008230452675] [294.61316872427983, 164.43621399176953] if stre is 0.65
#[258.2078189300411, 243.98765432098764] [432.46090534979425, 115.059670781893] if actu is 0.2
#[413.9794238683128, 270.3333333333333] [687.4958847736625, 128.38683127572017]if sens is 0.2

#==> exp exh stre and actu change things
#[261.0925925925926, 283.31481481481484] [530.8333333333334, 142.00617283950618]if exp 0
#[449.679012345679, 275.8744855967078] [728.7057613168724, 134.39506172839506] if exp 1
#[531.7201646090535, 272.24074074074076] [774.7448559670781, 128.85802469135803] if exp is 2

#[122.14197530864197, 278.5576131687243] [195.28600823045267, 141.559670781893]if exh is 0
#[593.0267489711935, 276.10493827160496] [944.2304526748972, 128.6851851851852] if exh is 1
#[527.3230452674898, 276.76748971193416] [894.7674897119341, 135.01440329218107]if exh is 2

#[393.3374485596708, 191.84567901234567] [702.8600823045267, 89.18312757201646]if stre is 0.25
#[622.4711934156379, 309.4835390946502] [1036.8106995884773, 151.63991769547326] if stre is 0.45
#[226.68312757201647, 330.1008230452675] [294.61316872427983, 164.43621399176953] if stre is 0.65

#[585.366255144033, 307.38271604938274] [979.7283950617284, 151.65843621399176] if actu is 0.05
#[398.917695473251, 280.059670781893] [622.0946502057614, 138.5411522633745] if actu is 0.1
#[294.55864197530866, 199.78703703703704] [510.7746913580247, 93.05555555555556] if actu is 0.2

##=> higher exp means higher analog threshold hits
##=> higher exhaustion means less non anlog threshold hits and middle means higher analog threshold hits
##=>strength not complete yet
##=> higher actu means less threshold hits

##stre and exh are closely connected 
#%%
import glob
import pandas as pd
import numpy as np

# Get the list of all CSV files
all_files = glob.glob("./gridResCSVNew_31_03_25_*.csv")

# Load all CSV files into DataFrames
data_frames = [pd.read_csv(file) for file in all_files]

# Assuming all DataFrames have the same common columns for the first data frame
common_columns = ['exh', 'exp', 'pred', 'mem', 'stre', 'actu', 'sens']

# Extract common columns from the first file (since they are identical across all files)
common_data = data_frames[0][common_columns]

# Define the function to split columns
def split_column(df, column_name):
    return df[column_name].str.strip('[]').str.split(',', expand=True).astype(float)

# Initialize lists to store split data
NAt_A_list = []
NAt_N_list = []
At_A_list = []
At_N_list = []

# Process each DataFrame to split NAt and At columns and collect split parts
for df in data_frames:
    NAt_split = split_column(df, 'NAt')
    At_split = split_column(df, 'At')
    
    NAt_A_list.append(NAt_split[0])
    NAt_N_list.append(NAt_split[1])
    At_A_list.append(At_split[0])
    At_N_list.append(At_split[1])

# Calculate medians for each new column
n_rows = data_frames[0].shape[0]

median_data = {
    'NAt_A': [np.median([NAt_A_list[i][j] for i in range(len(NAt_A_list))]) for j in range(n_rows)],
    'NAt_N': [np.median([NAt_N_list[i][j] for i in range(len(NAt_N_list))]) for j in range(n_rows)],
    'At_A': [np.median([At_A_list[i][j] for i in range(len(At_A_list))]) for j in range(n_rows)],
    'At_N': [np.median([At_N_list[i][j] for i in range(len(At_N_list))]) for j in range(n_rows)],
}

# Create a DataFrame for median values
median_df = pd.DataFrame(median_data)

# Combine the common columns with median data
final_data = pd.concat([common_data, median_df], axis=1)

# Save the final combined DataFrame to a new CSV file
final_data.to_csv("combined_median.csv", index=False)
