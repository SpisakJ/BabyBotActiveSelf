#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:26:00 2023

@author: spisak
"""
#%%
#import torch and pytorch_lightning for machine learning
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
from Pressure2Sound_JH import Pressure2Soundv2
from Pressure2Sound_JH import Pressure2Soundv3
from Pressure2Sound_JH import Pressure2Soundv3

#importing numpy for some maths such as sinus curve
import numpy as np

#import matplotlib for graph plotting
import matplotlib.pyplot as plt
#%%
#dataset that produces dummies as we need it for the torch lightning module
steps = 5400
class Infinite(Dataset):

    def __len__(self):
        return steps

    def __getitem__(self, idx):
        return torch.rand(1)
    
#model for oral activity takes in past sound to produce new force
class oralActivity(torch.nn.Module):
    def __init__(self,memory):
        super(oralActivity,self).__init__()
        self.memory = memory
        self.linear = torch.nn.Linear(self.memory , self.memory+1) #first layer takes in past sounds
        self.linear2 = torch.nn.Linear(self.memory+1, 1) #second layer produces new sound
        self.sigmoid = torch.nn.Sigmoid()
        
        #we initilise our layers with ones for their weights. This serves two purposes:
            #every model has the exact same start
            #we start high so that we do not get trapped in local minimas where we just produce zeros
        self.linear.weight = torch.nn.init.ones_(self.linear.weight) 
        self.linear2.weight = torch.nn.init.ones_(self.linear2.weight) 
        
    def forward(self,pastSounds):
        x = self.linear(pastSounds)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x,pastSounds
    
#our model for sound prediction from force
class soundPrediction(torch.nn.Module):
    def __init__(self):
        super(soundPrediction,self).__init__()
        self.linear = torch.nn.Linear(1,10) #takes in force
        self.linear2 = torch.nn.Linear(10,1) #outputs  predicted sound
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.linear(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
class  BabyBot(pl.LightningModule):
    def __init__(self,suckingFrequency = 0.1, memory = 10, noise=0.1, lr=0.001, condition="analog",threshold=0.1,age="old",sensoryNoise = 0.1,lossWeigths=[1.0,1.0,1.0],strength=0.4,device="cuda",device="cuda"):
        super(BabyBot,self).__init__()
        #age of the baby only used for plotting
        self.age= age
        
        #threshold under which no sound is produced
        self.threshold = threshold
        
        #noise towards how accurate the baby controls its force
        self.noise = noise
        
        #learning rate of our model 
        self.lr = lr
        
        #start time
        self.time = 0
        
        #how time is progressed
        self.fq = suckingFrequency
        #self.device = "cuda"
        
        #how much past sounds are rembembered
        self.memory = memory
        
        #what the experimental condition is
        self.condition = condition
        
        #the past sound, we start with a random value between 0 and 1
        self.pastSounds = torch.rand(memory,requires_grad=True).float().to(self.device).to(self.device)
        
        #our oral activity mdoel
        self.oralActivity = oralActivity(self.memory)
        
        #our sound prediction model
        self.soundPredictionModel = soundPrediction()
        
        #mean square error loss
        self.mseLoss = torch.nn.MSELoss()
        
        #tThis is badly named, i use it to keep track of which condition was used at which timestep.
        self.label = []
        
        #variable to record losses
        self.recLosses = []
        self.forces = []
        #variable to record produced sounds
        self.producedSoundsT = []
        
        #variable to record produced sounds
        self.wantedForcesT = []
        
        #variable to record produced force
        self.producedForcesT = []
        
        #strenght of the baby, influences instinct, this means it is very important for our baseline
        self.strength = strength
        
        #another noise we call sensory noise which influenenced how correctly the baby heard the past sound
        self.sensoryNoise = sensoryNoise

        self.theX =[]
        self.theFreq = []
        
        #how long we train on one condition
        self.conditionLength = 5400*1/6
        
        #to weigh losses differently
        self.lossWeigths = lossWeigths
        
        #this is so that in non anlog condition next timesteps after being over threshold still make sound
        self.leftOverSoundCounter = 100
        
        self.pacifier = Pressure2Soundv3.Pacifier()
        
        self.pacifier = Pressure2Soundv3.Pacifier()
        
    def mapToReal(self):
        #
        return self.producedForcesT#pacifier.map_pressure_to_frequency(self.producedForcesT, len(self.producedForcesT)/10)
    
    def realBack(self,desiredPressure):
            
        if self.time*10 < self.conditionLength:
            return torch.tensor(0)
        if self.time*10 > self.conditionLength*5:
            return torch.tensor(0)
        self.theX.append(self.pacifier.x)
        self.theFreq.append(self.pacifier.frequency)
        self.pacifier.run(desiredPressure,self.condition,1)
        return self.pacifier.x#frequency#x#frequency#x[0]#frequency
        
    #models time passed important for instinct
    def timestep(self):
        self.time += self.fq 
        
        
    #an instinct towards sucking here modeled with a sinus function multiplied by a strength factor
    def instinct(self):
        return torch.tensor([(np.sin(self.time)+1)/2]).double().to(self.device)*self.strength#between 0 and 1, using sinus function not sure if fitting for instinct better functions might exist
        return torch.tensor([(np.sin(self.time)+1)/2]).double().to(self.device)*self.strength#between 0 and 1, using sinus function not sure if fitting for instinct better functions might exist
    
    #function that uses our model for oral activity
    def regulateSucking(self):
        #the model has the last sound as its input
        producedForce = self.oralActivity(self.pastSounds.to(self.device).to(self.device))
        return producedForce
    
    #updates past sound
    def addSoundToPastSounds(self,sound):
        pastSounds = self.pastSounds[1:]
        pastSounds = torch.cat((pastSounds,sound)).to(self.device).to(self.device)
        self.pastSounds = pastSounds
        
    #calculates sound from force. im only using this during eval as it made the tracing harder. It should be adjusted so it can be used in train step as well    
    def forceToSound(self,force):
        if self.condition == "analog":
            if force > self.threshold: #og paper has a treshold of some force not sure hwo to translate the amount of force
                sound = force #sound depends on force
            else:
                sound = torch.tensor([0.0],requires_grad=True).float().to(self.device)
        elif self.condition == "non-analog":
                sound = torch.tensor([0.0],requires_grad=True).float().to(self.device)
        elif self.condition == "non-analog":
            if force > self.threshold:
                sound = force*0+torch.rand(1).float().to(self.device).to(self.device) #sound does not depend on force
            else:
                sound = torch.tensor([0.0],requires_grad=True).float().to(self.device).to(self.device)
        sound = torch.tensor([sound]).to(self.device).to(self.device)
        self.addSoundToPastSounds(sound)
        return sound
    #uses the prediction model to predict sound from current force
    def soundPrediction(self,regulatedForce):
        predictedSound = self.soundPredictionModel(regulatedForce)
        return predictedSound
    #optimizers for the networks
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    #to load data, this is a required function for torch lighnting modules the data is not used  so its mostly a dummy
    def train_dataloader(self):
        training_data = Infinite()
        train_loader = DataLoader(training_data, num_workers=0, batch_size = 1, shuffle=False)
        return train_loader
    
    # adds noise to the force we get a random value multiply it by 2 and subtract it from 1 to get values between -1 and 1
    def suckingNoise(self,force): #this models the inaccuracy of the baby in regulating sucking force
        return force + (1-torch.rand(1).to(self.device).to(self.device)*2)*self.noise
    
    #this function does the actual training
    def training_step(self,batch):
        
        # we want the current instinct
        instinctForce = self.instinct()
        #advance time so next instinct is differnt
        self.timestep()
        
        #use model to gain current force, the model does take in the last sound but here we just use the function from this class
        #ps is past sound
        regulatedForce,ps = self.regulateSucking()
        
        self.wantedForcesT.append(regulatedForce.item())
        
        #we predict a sound using the prediction model
        predictedSound = self.soundPrediction(regulatedForce.to(self.device).to(self.device))
        
        #we add noise with the noise function
        regulatedForce = self.suckingNoise(regulatedForce)
        
        regulatedForce2 = self.realBack(regulatedForce.item())
        #print(regulatedForce2,regulatedForce.item())
        
        self.forces.append(regulatedForce.item())
        
        #record produced force
        self.producedForcesT.append(regulatedForce2/400)
        
        #we calculate exhaustion which is how different the sucking force is from the instinct, perhaps this should only work if its higher than instinct 
        exhaustion = self.mseLoss(regulatedForce.to(self.device).to(self.device),instinctForce.float().to(self.device).to(self.device))#abs(regulatedForce-instinctForce)
        
        #TODO change it so we have baseline as another condition
        #here I want to change the condition based on time to passed to be closer to the original experiement where we have baseline->analog->nonanalog and so on in sequence
        #here I want to change the condition based on time to passed to be closer to the original experiement where we have baseline->analog->nonanalog and so on in sequence
        if self.time*10 > self.conditionLength*2: #*2 because first baseline and then the start condition
            if (self.time*10 -self.conditionLength)% self.conditionLength <= 1:
                if self.condition=="analog":
                    self.condition= "non-analog"
                elif self.condition=="non-analog":
                    self.condition= "non-analog"
                elif self.condition=="non-analog":
                    self.condition="analog"
        
        #basicly the force to sound function but adjusted and copied here to make back propagation easier 
        #this should be changed so that it just uses the existing funciton
        if self.time*10 < self.conditionLength or self.time*10 > 5* self.conditionLength:
            regulatedForce = regulatedForce*0
        
        elif self.condition == "analog":
            if regulatedForce > self.threshold: #og paper has a treshold of some force not sure hwo to translate the amount of force
                regulatedForce = regulatedForce #sound depends on force
            else:
                regulatedForce = regulatedForce*0#torch.tensor([0.0],requires_grad=True).float().cuda()
        elif self.condition == "non-analog":
        elif self.condition == "non-analog":
            if regulatedForce > self.threshold:
                regulatedForce = regulatedForce*0+torch.rand(1).float().to(self.device).to(self.device) #sound does not depend on force
                self.leftOverSoundCounter = 100
            else:
                if self.leftOverSoundCounter <= 0:
                    regulatedForce = regulatedForce*0#torch.tensor([0.0],requires_grad=True).float().cuda()
                else:
                    regulatedForce = regulatedForce*0+torch.rand(1).float().to(self.device) 
                    self.leftOverSoundCounter -= 1
        
        #now we know the true sound which is still called regulated force here and find the prediction loss between it and our predicted sound
        predictionLoss = self.mseLoss(predictedSound,torch.tensor(torch.tensor(regulatedForce2/400).float().to(self.device)2/400).float().to(self.device))
        
        #if len(ps) > 1:
        #    ps = ps[-1]
        #here we want to create a label as far away from our last sound as possible to encourage exploration in our used force
        if ps[-1] < 0.5:
            label = ps[-1]**0
        elif ps[-1] >= 0.5:
            label= ps[-1]*0
            
        #using the label just created we want to minimize the distance from it to our current sound
        repLoss = 1-self.mseLoss(regulatedForce,ps[-1])
        
        #we add all our losses together
        loss =repLoss*self.lossWeigths[0]+predictionLoss*self.lossWeigths[1]+exhaustion*self.lossWeigths[2]
        
        #we record the loss which can be used to display the loss after training
        self.recLosses.append(loss.item())
        
        #we update the past sound with the current sound
        with torch.no_grad():
            self.pastSounds = torch.cat((self.pastSounds,torch.tensor([regulatedForce.item()])+self.sensoryNoise))
            self.pastSounds  = self.pastSounds[1:]
        #self.pastSounds = torch.tensor([regulatedForce.item()]).cuda()+self.sensoryNoise
        #we record the current sound so that we can display the sounds created during training at a later time.
        self.producedSoundsT.append(regulatedForce2.item()/400)
        if self.condition == "analog":
            self.label.append(1)
        else:
            self.label.append(0)
        
        
        return loss
    #this function does an eval run on it and produces some graphs so we can see how the sound and force looks at the end
    def showBehaviour(self):        
        #variable to keep the amplitudes of sound in eval, was the amplitude in the paper for force or sound? would need to be adjusted if its force
        amplitude =[]
            
        #variable to count no sound happening in eval
        zeroCount=0
        thresholdCountAna = 0
        thresholdCountNAna = 0
        #go through produced sounds
        for sound in self.producedSoundsT:
            
            #how far away from the average sound is the current sound
            amp = abs(np.average(self.producedSoundsT)-sound)
            
            #add it to our amplitude variable
            amplitude.append(amp)
            
            #if sound is 0 add 1 to our zero count
            if sound ==0:
                zeroCount+=1
        for i in range (len( self.theX)):
            
            if self.theX[i] >= 0.1667 and self.theX[i]<= 0.2083:
                if self.label[i] ==1:
                    thresholdCountAna +=1
                elif self.label[i] == 0:
                    thresholdCountNAna +=1
        return np.average(self.producedSoundsT),np.average(amplitude),zeroCount,[thresholdCountAna,thresholdCountNAna]
    
    
'''
This is the execution, you can change the variables in myModel = BabyBot()
novelty does not really do anything right now
condition can be anloag or non-analog and decides on how force is translated into sound.
condition can be anloag or non-analog and decides on how force is translated into sound.
noise is how accurate the model can choose the force so its added before the force is translated to sound
lr is the learning rate of the model
memory is how far in the past we remeber the sounds, currently this should stay at 1 or it might not function, There also should not be any 
    advantage to a higher value as the only past sound that should matter to us is the last one witht he way the loss is set up
    this could be changed of course in order to increase the exploration in some ways so that we want values further from the average of the past
    x sounds
threshold is the value at which the force starts to make sound so forces below this wont make any sound at all
age is named quietly badly and should probably be changed at some point, right now it decides on how many layers we want to have in the model
    that decides onhow much force we are going to use next.,fo
'''

#sensor noise

#loss weighted
import time
import pandas as pd
#function to run experiments
def runExperiment(epochs,condition,noise,learningRate,memory,threshold,runs,age,sensorNoise,lossWeights,strength):
    #variables to record results
    averageSound = []
    averageAmp = []
    zeros = []
    thresholdHit =[]
    noiseRange =np.random.uniform(low=noise,high=noise,size=runs) #np.random.uniform(low=noise-0.1,high=noise+0.1,size=runs)
    sensoryNoiseRange = np.random.uniform(low=sensorNoise,high=sensorNoise,size=runs)#np.random.uniform(low=sensorNoise-0.1,high=sensorNoise+0.1,size=runs)
    #how often we want to run this condition 
    for i in range(runs):
        
        #sets up the babybot, here the parameters are decided
        myModel = BabyBot(condition=condition,noise=noiseRange[i],lr=learningRate,memory=memory,threshold=threshold,age=age,sensoryNoise=sensoryNoiseRange[i],lossWeigths=lossWeights,strength=strength)
        trainer = pl.Trainer("cpu",max_epochs=epochs)
        now = time.time()
        trainer.fit(model=myModel)   
        after = time.time()
        print(after-now)
        
        #call funciton of the model to disply some results
        avg,amp,z,t = myModel.showBehaviour()    
        freq = myModel.theX#theFreq#theX#wantedForcesT
        nfreq =[]
        for f in range(len(freq)):
            fi = f
            f = freq[fi]
            try:
                    nfreq.append( f.item())
            except:
                    nfreq.append( f)


                    print("go")

        print(myModel.theX)
        #record results so that we can create the average of the runs 
        averageSound.append(avg)
        averageAmp.append(amp)
        zeros.append(z)
        thresholdHit.append(t)
        df = pd.DataFrame(myModel.forces)
        df.to_csv(f"run_{i}_start_{condition}_age_{age}.csv",index=None,header=None)
    return ["averageSound",np.average(averageSound),"averageAmplitude",np.average(averageAmp),"averageZeros",np.average(zeros),"TresholdHits [Analog,nonAnalog]",thresholdHit],freq
import time
def gridSearch():
    startTime = time.time()
    exploreLoss =  [0,1,2]
    exhaustionLoss = [0,1,2]
    predictionLoss = [0,1,2]
    memory  = [1,5]
    strength = [0.25,0.45,0.65]
    actuaryNoise  = [0.05, 0.1, 0.2]
    sensoryNoise  = [0.05, 0.1, 0.2]
    results = []
    for exp in exploreLoss:
        for exh in exhaustionLoss:
            for pred in predictionLoss:
                for mem in memory:
                    for stre in strength:
                        for actu in actuaryNoise:
                            for sens in sensoryNoise:
                                startTime = time.time()
                                myModelAnaSt = BabyBot(condition ="analog",noise = actu,lr = 0.005,memory=mem,threshold = 0.1667,age = "old",sensoryNoise=sens,lossWeigths=[exp,pred,exh],strength=stre) 
                                trainer = pl.Trainer("gpu",max_epochs=1,logger=False)
                                trainer.fit(model=myModelAnaSt)
                                aavg,aamp,az,at = myModelAnaSt.showBehaviour()
                                afreq = myModelAnaSt.mapToReal()
                                
                                myModelNAnaSt = BabyBot(
                                    condition ="non-analog",
                                    noise = actu,
                                    lr = 0.005,
                                    memory=mem,
                                    threshold = 0.1667,
                                    age = "old",
                                    sensoryNoise=sens,
                                    lossWeigths=[exp,pred,exh],
                                    strength=stre
                                    ) 
                                trainer = pl.Trainer("gpu",max_epochs=1,logger=False)
                                trainer.fit(model=myModelNAnaSt)
                                avg,amp,z,t = myModelNAnaSt.showBehaviour()
                                freq = myModelNAnaSt.mapToReal()
                                params = {"exh":exh,"exp":exp,"pred":pred,"mem":mem,"stre":stre,"actu":actu,"sens":sens,
                                          "NAavg":avg,"NAamp":amp,"NAz":z,"NAt":t,"NAfreq":freq,
                                          "Aavg":aavg,"Aamp":aamp,"Az":az,"At":at,"Afreq":afreq}
                                results.append(params)
                                endTime = time.time()
                                print(endTime-startTime)
    return results
def performAblations():
    baseParameters= {
        "actuaryNoise":0.005, # x4 young (not anymore)
        "learningRate":0.005,
        "memory":5,
        "threshold":0.1667,
        "runs":50,
        "age":"young",
        "sensoryNoise":0.05,
        "lossWeights1":1, # 0 1 1 
        "lossWeights2":1,
        "lossWeights3":1,
        "strength":0.25}
    for key in baseParameters.keys():   
        orKeyValue = baseParameters[key]
        baseParameters[key] = 0
        analogRuns,analogFr = runExperiment(1,
                        "analog", 
                        baseParameters["actuaryNoise"],
                        baseParameters["learningRate"],
                        baseParameters["memory"],
                        0.1667,
                        baseParameters["runs"],
                        "old",
                        baseParameters["sensoryNoise"],
                        [baseParameters["lossWeights1"],
                         baseParameters["lossWeights2"],
                         baseParameters["lossWeights3"]],
                        baseParameters["strength"])
        nonAnalogRuns,nonAnalogFR = runExperiment(1,
                        "non-analog", 
                        baseParameters["actuaryNoise"],
                        baseParameters["learningRate"],
                        baseParameters["memory"],
                        0.1667,
                        baseParameters["runs"],
                        "old",
                        baseParameters["sensoryNoise"],
                        [baseParameters["lossWeights1"],
                         baseParameters["lossWeights2"],
                         baseParameters["lossWeights3"]],
                        baseParameters["strength"])
        
        fileNameAnalog  = f"ablations_start_analog_steps_{steps}_actuaryNoise_{baseParameters['actuaryNoise']}_learningRate_{baseParameters['learningRate']}_memory_{baseParameters['memory']}_threshold_{baseParameters['threshold']}_runs_{baseParameters['runs']}_age_{baseParameters['age']}_sensoryNoise_{baseParameters['sensoryNoise']}_lossWeights1_{baseParameters['lossWeights1']}_lossWeights2_{baseParameters['lossWeights2']}_lossWeights3_{baseParameters['lossWeights3']}_strength_{baseParameters['strength']}.csv"
        fileNameNonAnalog  = f"ablations_start_non-analog_steps_{steps}_actuaryNoise_{baseParameters['actuaryNoise']}_learningRate_{baseParameters['learningRate']}_memory_{baseParameters['memory']}_threshold_{baseParameters['threshold']}_runs_{baseParameters['runs']}_age_{baseParameters['age']}_sensoryNoise_{baseParameters['sensoryNoise']}_lossWeights1_{baseParameters['lossWeights1']}_lossWeights2_{baseParameters['lossWeights2']}_lossWeights3_{baseParameters['lossWeights3']}_strength_{baseParameters['strength']}.csv"
        df = pd.DataFrame(analogRuns)
        df.to_csv(fileNameAnalog,index=None,header=None)
        df = pd.DataFrame(nonAnalogRuns)
        df.to_csv(fileNameNonAnalog,index=None,header=None)
        baseParameters[key] = orKeyValue
        print(f"Finished {key}")
performAblations()      
#gridResults = gridSearch()
#import pandas as pd
#df = pd.DataFrame(gridResults)
#df.to_csv("./gridResCSVNew_31_03_25.csv")
#%%
'''
#non analog young baby experiment now only starts with nonanalog
#BaseValues: everything that is different means a change for the experiement, should always be the same for both runs of one age except for condition
epochs = 1
strength = 0.25
memory = 1 #*5 old
learningRate = 0.005
threshold = 0.1667
actuaryNoise = 0.5 # x4 young (not anymore)
sensoryNoise = 0.05
runs = 10
lossWeights = [1.,1.,1.0] # 0 1 1 young
# params: epochs,condition,noise,learningRate,memory,threshold,runs,age,sensorNoise,lossWeights,strength
nonAnalogYoung,fr1  = runExperiment(epochs,"non-analog", actuaryNoise, learningRate,memory,threshold,runs,"young",sensoryNoise,[0.0,1.0,1.0],strength)
# analog young baby experiment
#analogYoung,fr2 = runExperiment(epochs,"analog", actuaryNoise, learningRate,memory,threshold,runs,"young",sensoryNoise,[0.0,1.0,1.0],strength)
#non analog old baby experiment
nonAnalogOld,fr3 = runExperiment(epochs,"non-analog", actuaryNoise, learningRate,memory*5,threshold,runs,"old",sensoryNoise,lossWeights,strength)
# analog old baby experiment
#analogOld,fr4 = runExperiment(epochs,"analog", actuaryNoise, learningRate,memory*5,threshold,runs,"old",sensoryNoise,lossWeights,strength)
'''
'''
mem 2 instead of 5
Old Baby, nonAnalog Start:  ['averageSound', 0.00017879840908878653, 'averageAmplitude', 0.00014459017205667018, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[46, 604], [17, 379], [3, 360], [21, 82], [13, 75], [16, 584], [1, 7], [24, 493], [17, 59], [4, 39]]]
Old Baby Analog Start:   ['averageSound', 0.00018169202614769887, 'averageAmplitude', 0.0001406647298977986, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[52, 1], [34, 3], [57, 47], [13, 5], [46, 9], [151, 170], [263, 269], [17, 5], [45, 14], [305, 352]]]
Young Baby nonAnalog Start:  ['averageSound', 0.0001741648592872641, 'averageAmplitude', 0.00012455691724444512, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[1, 27], [26, 42], [6, 26], [2, 12], [6, 14], [21, 45], [3, 12], [15, 42], [17, 46], [10, 39]]]
Young Baby Analog Start: ['averageSound', 0.0001384728425712361, 'averageAmplitude', 0.00010986007199659028, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[45, 6], [16, 5], [128, 19], [40, 7], [19, 6], [47, 22], [21, 1], [5, 1], [27, 7], [43, 11]]]
no explo diff
Old Baby, nonAnalog Start:  ['averageSound', 0.00016988175008200175, 'averageAmplitude', 0.00013148138968779128, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[2, 6], [19, 32], [31, 251], [34, 547], [1, 14], [5, 28], [8, 35], [11, 96], [1, 357], [3, 22]]]
Old Baby Analog Start:   ['averageSound', 0.00017449387280519244, 'averageAmplitude', 0.0001332153783936728, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[79, 18], [25, 11], [153, 259], [114, 257], [68, 4], [14, 5], [9, 3], [10, 0], [91, 85], [44, 19]]]
Young Baby nonAnalog Start:  ['averageSound', 0.00015717337706238835, 'averageAmplitude', 0.00012390094444003906, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[36, 685], [0, 29], [6, 152], [0, 13], [4, 344], [2, 20], [16, 69], [16, 145], [18, 46], [19, 58]]]
Young Baby Analog Start: ['averageSound', 0.00018740552650638043, 'averageAmplitude', 0.00014340715701322031, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[49, 44], [178, 381], [30, 11], [13, 0], [53, 139], [22, 14], [7, 0], [296, 255], [439, 358], [11, 0]]]

high noise
Old Baby, nonAnalog Start:  ['averageSound', 0.00013711107598005432, 'averageAmplitude', 0.00013632625788951067, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[13, 65], [19, 138], [11, 76], [23, 90], [19, 90], [17, 139], [8, 98], [11, 84], [9, 134], [16, 205]]]
Old Baby Analog Start:   ['averageSound', 0.00037556021025325864, 'averageAmplitude', 0.00048462045229242037, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[27, 77], [16, 67], [57, 42], [45, 30], [71, 55], [46, 20], [19, 87], [29, 43], [27, 87], [9, 69]]]
Young Baby nonAnalog Start:  ['averageSound', 9.281916769257824e-05, 'averageAmplitude', 8.842510225588057e-05, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[22, 43], [19, 82], [24, 53], [19, 49], [17, 74], [15, 54], [16, 35], [21, 50], [20, 58], [19, 63]]]
Young Baby Analog Start: ['averageSound', 9.25533993515123e-05, 'averageAmplitude', 8.860664030002378e-05, 'averageZeros', 1801.0, 'TresholdHits [Analog,nonAnalog]', [[66, 25], [48, 26], [34, 14], [54, 17], [60, 26], [46, 11], [61, 15], [46, 27], [46, 13], [84, 28]]]

'''


#prints of the results
'''
print("Old Baby, nonAnalog Start: ",nonAnalogOld)
print("Old Baby Analog Start:  ",analogOld)      
print("Young Baby nonAnalog Start: ",nonAnalogYoung)
print("Young Baby Analog Start:",analogYoung)  
#%%
import pandas as pd
df = pd.read_csv("run_3_start_non-analog_age_old.csv")
values = df.values
fr3 = values
#%%
def flatter (data):
    return  [float(val) if isinstance(val, np.ndarray) else val for val in data]
#%%
data = np.array(flatter(fr1))
sns.scatterplot(x=range(len(data)),y=data,s=5).set_title("non-analog start, young")
sns.kdeplot(x=range(len(data)),y=data,fill=True,alpha=0.7)
#%%

data = np.array(flatter(fr2))
sns.scatterplot(x=range(len(data)),y=data,s=5).set_title("analog start, young")
sns.kdeplot(x=range(len(data)),y=data,fill=True,alpha=0.7)
#%%
data = np.array(flatter(fr3))
sns.scatterplot(x=range(len(data)),y=data,s=5).set_title("non-analog start, old")
sns.kdeplot(x=range(len(data)),y=data,fill=True,alpha=0.7)
#%%
data = np.array(flatter(fr4))
sns.scatterplot(x=range(len(data)),y=data,s=5).set_title("analog start, old")
sns.kdeplot(x=range(len(data)),y=data,fill=True,alpha=0.7) 
#%%
#plt.xlabel("time steps")
#plt.ylabel("intensity")
#plt.plot(fr1,"ob", markersize=0.5)
#plt.show()
#plt.xlabel("time steps")
#plt.ylabel("intensity")
#plt.plot(fr2,"ob", markersize=0.5)
#plt.show()
#plt.xlabel("time steps")
#plt.ylabel("intensity")
#plt.plot(fr3,"ob", markersize=0.5)
#plt.show()
#plt.xlabel("time steps")
#plt.ylabel("intensity")
#plt.plot(fr4,"ob", markersize=0.5)#
#plt.show()
'''