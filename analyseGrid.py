import pandas as pd
import numpy as np
from ast import literal_eval
df = pd.read_csv("Combined.csv")

df2 = df[["exp","pred","exh","mem","stre","actu","sens","NAt","NNt","AAt","ANt"]]

print(df2.corr(min_periods=3))
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
for rowA,rowN in zip(df.loc[(df["exp"].isin(exp)) & (df["pred"].isin(pred)) & (df["exh"].isin(exh)) & (df["mem"].isin(mem)) & (df["stre"].isin(stre)) & (df["actu"].isin(actu)) & (df["sens"].isin(sens))]["NAt"] ,
                     df.loc[(df["exp"].isin(exp)) & (df["pred"].isin(pred)) & (df["exh"].isin(exh)) & (df["mem"].isin(mem)) & (df["stre"].isin(stre)) & (df["actu"].isin(actu)) & (df["sens"].isin(sens))]["AAt"]):
    count+=1
    avgA[0] += rowA[0]
    avgA[1] += rowA[1]
    avgN[0] += rowN[0]
    avgN[1] += rowN[1]
print(count)
print([avgA[0]/count,avgA[1]/count],[avgN[0]/count,avgN[1]/count])
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