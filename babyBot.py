#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:26:00 2023

@author: spisak
"""
#import torch and pytorch_lightning for machine learning
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

#importing numpy for some maths such as sinus curve
import numpy as np

#import matplotlib for graph plotting
import matplotlib.pyplot as plt

#dataset that produces dummies as we need it for the torch lightning module
class Infinite(Dataset):

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.rand(1)
    
#model for oral activity takes in past sound to produce new force
class oralActivity(torch.nn.Module):
    def __init__(self,memory):
        super(oralActivity,self).__init__()
        self.memory = memory
        self.linear = torch.nn.Linear(1 , memory+1) #first layer takes in past sounds
        self.linear2 = torch.nn.Linear(memory+1, 1) #second layer produces new sound
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
    def __init__(self,suckingFrequency = 0.1, memory = 10, noise=0.1, lr=0.001, condition="analog",threshold=0.1,age="old",sensoryNoise = 0.1,lossWeigths=[1.0,1.0,1.0]):
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
        
        #how much past sounds are rembembered
        self.memory = memory
        
        #what the experimental condition is
        self.condition = condition
        
        #the past sound, we start with a random value between 0 and 1
        self.pastSounds = torch.rand(memory,requires_grad=True).float().cuda()
        
        #our oral activity mdoel
        self.oralActivity = oralActivity(self.memory).cuda()
        
        #our sound prediction model
        self.soundPredictionModel = soundPrediction()
        
        #mean square error loss
        self.mseLoss = torch.nn.MSELoss()
        
        #variable to record losses
        self.recLosses = []
        
        #variable to record produced sounds
        self.producedSoundsT = []
        
        #variable to record produced force
        self.producedForcesT = []
        
        #strenght of the baby, influences instinct, this means it is very important for our baseline
        self.strength = 0.4
        
        #another noise we call sensory noise which influenenced how correctly the baby heard the past sound
        self.sensoryNoise = sensoryNoise
        
        #how long we train on one condition
        self.conditionLength = 2400*1/6
        
        #to weigh losses differently
        self.lossWeigths = lossWeigths
        
    #models time passed important for instinct
    def timestep(self):
        self.time += self.fq 
        
        
    #an instinct towards sucking here modeled with a sinus function multiplied by a strength factor
    def instinct(self):
        return torch.tensor([(np.sin(self.time)+1)/2]).double().cuda() *self.strength#between 0 and 1, using sinus function not sure if fitting for instinct better functions might exist
    
    #function that uses our model for oral activity
    def regulateSucking(self):
        #the model has the last sound as its input
        producedForce = self.oralActivity(self.pastSounds)
        return producedForce
    
    #updates past sound
    def addSoundToPastSounds(self,sound):
        pastSounds = self.pastSounds[1:]
        pastSounds = torch.cat((pastSounds,sound))
        self.pastSounds = pastSounds.cuda()
        
    #calculates sound from force. im only using this during eval as it made the tracing harder. It should be adjusted so it can be used in train step as well    
    def forceToSound(self,force):
        if self.condition == "analog":
            if force > self.threshold: #og paper has a treshold of some force not sure hwo to translate the amount of force
                sound = force #sound depends on force
            else:
                sound = torch.tensor([0.0],requires_grad=True).float()
        elif self.condition == "Non-analog":
            if force > self.threshold:
                sound = force*0+torch.rand(1).float().cuda() #sound does not depend on force
            else:
                sound = torch.tensor([0.0],requires_grad=True).float()
        sound = torch.tensor([sound]).cuda()
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
        return force + (1-torch.rand(1).cuda()*2)*self.noise
    
    #this function does the actual training
    def training_step(self,batch):
        
        # we want the current instinct
        instinctForce = self.instinct()
        #advance time so next instinct is differnt
        self.timestep()
        
        #use model to gain current force, the model does take in the last sound but here we just use the function from this class
        #ps is past sound
        regulatedForce,ps = self.regulateSucking()
        
        #record produced force
        self.producedForcesT.append(regulatedForce.item())
        
        #we predict a sound using the prediction model
        predictedSound = self.soundPrediction(regulatedForce)
        
        #we add noise with the noise function
        regulatedForce = self.suckingNoise(regulatedForce)
        
        #we calculate exhaustion which is how different the sucking force is from the instinct, perhaps this should only work if its higher than instinct 
        exhaustion = self.mseLoss(regulatedForce,instinctForce.float())#abs(regulatedForce-instinctForce)
        
        #here I want to change the condition based on time to passed to be closer to the original experiement where we have baseline->analog->Nonanalog and so on in sequence
        if self.time*10 > self.conditionLength:
            if (self.time*10 -self.conditionLength)% self.conditionLength == 0:
                if self.condition=="analog":
                    self.condition= "Non-analog"
                elif self.condition=="Non-analog":
                    self.condition="analog"
            
        
        
        #basicly the force to sound function but adjusted and copied here to make back propagation easier 
        #this should be changed so that it just uses the existing funciton
        if self.time*10 < self.conditionLength or self.time*10 > 5* self.conditionLength:
            regulatedForce = regulatedForce*0
        elif self.condition == "analog":
            if regulatedForce > self.threshold: #og paper has a treshold of some force not sure hwo to translate the amount of force
                regulatedForce = regulatedForce #sound depends on force
            else:
                regulatedForce = torch.tensor([0.0],requires_grad=True).float().cuda()
        elif self.condition == "Non-analog":
            if regulatedForce > self.threshold:
                regulatedForce = regulatedForce*0+torch.rand(1).float().cuda() #sound does not depend on force
            else:
                regulatedForce = torch.tensor([0.0],requires_grad=True).float().cuda()
        
        #now we know the true sound which is still called regulated force here and find the prediction loss between it and our predicted sound
        predictionLoss = self.mseLoss(predictedSound,regulatedForce)
        
        #here we want to create a label as far away from our last sound as possible to encourage exploration in our used force
        if ps < 0.5:
            label = ps**0
        elif ps >= 0.5:
            label= ps*0
            
        #using the label just created we want to minimize the distance from it to our current sound
        repLoss = self.mseLoss(regulatedForce,label)
        
        #we add all our losses together
        loss =repLoss*self.lossWeigths[0]+predictionLoss*self.lossWeigths[1]+exhaustion*self.lossWeigths[2]
        
        #we record the loss which can be used to display the loss after training
        self.recLosses.append(loss.item())
        
        #we update the past sound with the current sound
        self.pastSounds = torch.tensor([regulatedForce.item()]).cuda()+self.sensoryNoise
        
        #we record the current sound so that we can display the sounds created during training at a later time.
        self.producedSoundsT.append(regulatedForce.item())
        
        
        return loss
    #this function does an eval run on it and produces some graphs so we can see how the sound and force looks at the end
    def showBehaviour(self):
        
        #we want to record sounds and forces during eval
        producedSounds = []
        producedForces =[]
        
        #we are checking 100 timesteps
        for i in range(100):
            
            #set model to eval mode
            evalModel = self.oralActivity
            evalModel.eval()
            evalModel.cuda()
            
            #no need for backpropagation here
            with torch.no_grad():
                
                
                #check results from model
                regulatedForce,ps = evalModel(self.pastSounds) 
                
                #record produced force
                producedForces.append(regulatedForce.item())
                
                #add noise
                regulatedForce = self.suckingNoise(regulatedForce)
                
                #record true sound
                trueSound = self.forceToSound(regulatedForce)
                
                #item so that we have a value instead of a tensor
                trueSound = trueSound.item()
                
                #record sound
                producedSounds.append(trueSound)
                
                #update past sounds
                self.pastSounds = torch.tensor([trueSound]).cuda()
        
        #we plot the sounds and forces recorded here as well as the sound over the training
        plt.title("Eval "+self.condition +" " +self.age)
        plt.plot(producedSounds,"bo",markersize=2.1,label="sound")
        plt.plot(producedForces,"ro",markersize=2,label="force")
        plt.legend()
        plt.show()
        
        plt.title("Sound Train "+self.condition+" " + self.age)
        plt.plot(self.producedSoundsT,"bo",markersize=2.1,label="sound")
        plt.plot(self.producedForcesT,"ro",markersize=2,label="force")
        plt.legend()
        plt.show()
        
        #variable to keep the amplitudes of sound in eval, was the amplitude in the paper for force or sound? would need to be adjusted if its force
        amplitude =[]
            
        #variable to count no sound happening in eval
        zeroCount=0
        thresholdCount = 0
        #go through produced sounds
        for sound in producedSounds:
            
            #how far away from the average sound is the current sound
            amp = abs(np.average(producedSounds)-sound)
            
            #add it to our amplitude variable
            amplitude.append(amp)
            
            #if sound is 0 add 1 to our zero count
            if sound ==0:
                zeroCount+=1
        for force in producedForces:
            if force >= self.threshold-0.02 or sound<= self.threshold+0.02:
                thresholdCount +=1
        return np.average(producedSounds),np.average(amplitude),zeroCount,thresholdCount
    
    
'''
This is the execution, you can change the variables in myModel = BabyBot()
novelty does not really do anything right now
condition can be anloag or Non-analog and decides on how force is translated into sound.
noise is how accurate the model can choose the force so its added before the force is translated to sound
lr is the learning rate of the model
memory is how far in the past we remeber the sounds, currently this should stay at 1 or it might not function, There also should not be any 
    advantage to a higher value as the only past sound that should matter to us is the last one witht he way the loss is set up
    this could be changed of course in order to increase the exploration in some ways so that we want values further from the average of the past
    x sounds
threshold is the value at which the force starts to make sound so forces below this wont make any sound at all
age is named quietly badly and should probably be changed at some point, right now it decides on how many layers we want to have in the model
    that decides onhow much force we are going to use next.
'''

#sensor noise

#loss weighted

#function to run experiments
def runExperiment(epochs,condition,noise,learningRate,memory,threshold,runs,age,sensorNoise):
    #variables to record results
    averageSound = []
    averageAmp = []
    zeros = []
    thresholdHit =[]
    
    noiseRange = np.random.uniform(low=noise-0.1,high=noise+0.1,size=runs)
    sensoryNoiseRange = np.random.uniform(low=sensorNoise-0.1,high=sensorNoise+0.1,size=runs)
    #how often we want to run this condition 
    for i in range(runs):
        
        #sets up the babybot, here the parameters are decided
        myModel = BabyBot(condition=condition,noise=noiseRange[i],lr=learningRate,memory=memory,threshold=threshold,age=age,sensoryNoise=sensoryNoiseRange[i])
        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(model=myModel)   
        
        #call funciton of the model to disply some results
        avg,amp,z,t = myModel.showBehaviour()    
        
        #record results so that we can create the average of the runs 
        averageSound.append(avg)
        averageAmp.append(amp)
        zeros.append(z)
        thresholdHit.append(t)
    return np.average(averageSound),np.average(averageAmp),np.average(zeros),np.average(thresholdHit)

#non analog young baby experiment now only starts with nonanalog
nonAnalogYoungSound, nonAnalogYoungAmp, nonAnalogYoungZeros,nonAnalogYoungT = runExperiment(2400,"Non-analog", 0.2, 0.01,1,0.16,2,"young",0.2)
# analog young baby experiment
analogYoungSound, analogYoungAmp, analogYoungZeros,analogYoungT = runExperiment(2400,"analog", 0.2, 0.01,1,0.16,2,"young",0.2)
#non analog old baby experiment
nonAnalogOldSound, nonAnalogOldAmp, nonAnalogOldZeros,nonAnalogOldT = runExperiment(2400,"Non-analog", 0.1, 0.01,1,0.16,2,"old",0.1)
# analog old baby experiment
analogOldSound, analogOldAmp, analogOldZeros,analogOldT = runExperiment(2400,"analog", 0.1, 0.01,1,0.16,2,"old",0.1)

#prints of the results
print("Old Baby: NonAnalog average, amplitude and zeros and thresholdHits:",nonAnalogOldSound, nonAnalogOldAmp, nonAnalogOldZeros,nonAnalogOldT)
print("Old Baby: Analog average, amplitude and zeros and thresholdHits:",analogOldSound, analogOldAmp, analogOldZeros,analogOldT)      
print("Young Baby: NonAnalog average, amplitude and zeros and thresholdHits:",nonAnalogYoungSound,nonAnalogYoungAmp,nonAnalogYoungZeros,nonAnalogYoungT)
print("Young Baby: Analog average, amplitude and zeros and thresholdHits:",analogYoungSound, analogYoungAmp, analogYoungZeros,analogYoungT)    


#TODO BASE ANA NA ANA NA BA
#THRESHOLD 0.1 psi 0.6 max
#%% old and young are based on the number of epochs we train for, 10 times as much trainig for old. 5 runs for each mode
'''
Old Baby: NonAnalog average, amplitude and zeros: 0.07111799454689025 0.12020740448951721 85.0
Old Baby: Analog average, amplitude and zeros: 0.3766379394829273 0.11275526656568051 4.8
Young Baby: NonAnalog average, amplitude and zeros: 0.3813555111885071 0.27084537514448165 25.4
Young Baby: Analog average, amplitude and zeros: 0.4621373709738254 0.109593042871356 2.2

#here i devided them by learning rate instead, noise is also a bit higher
Old Baby: NonAnalog average, amplitude and zeros: 0.2543489261865616 0.26236681587457655 50.8
Old Baby: Analog average, amplitude and zeros: 0.279271654009819 0.18987583573400973 33.4
Young Baby: NonAnalog average, amplitude and zeros: 0.440678873181343 0.26595384658336635 14.8
Young Baby: Analog average, amplitude and zeros: 0.6500598652660847 0.17061184880316257 6.2
'''