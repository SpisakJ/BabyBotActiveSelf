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



#importing numpy for some maths such as sinus curve

import numpy as np



#import matplotlib for graph plotting

import matplotlib.pyplot as plt



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

    def __init__(self,suckingFrequency = 0.1, memory = 10, noise=0.1, lr=0.001, condition="analog",threshold=0.1,age="old",sensoryNoise = 0.1,lossWeigths=[1.0,1.0,1.0],strength=0.4,device="cuda"):

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

        self.pastSounds = torch.rand(memory,requires_grad=True).float().to(self.device)



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



    def mapToReal(self):
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



    #function that uses our model for oral activity

    def regulateSucking(self):

        #the model has the last sound as its input

        producedForce = self.oralActivity(self.pastSounds.to(self.device))

        return producedForce



    #updates past sound

    def addSoundToPastSounds(self,sound):

        pastSounds = self.pastSounds[1:]

        pastSounds = torch.cat((pastSounds,sound)).to(self.device)

        self.pastSounds = pastSounds



    #calculates sound from force. im only using this during eval as it made the tracing harder. It should be adjusted so it can be used in train step as well    

    def forceToSound(self,force):

        if self.condition == "analog":

            if force > self.threshold: #og paper has a treshold of some force not sure hwo to translate the amount of force

                sound = force #sound depends on force

            else:

                sound = torch.tensor([0.0],requires_grad=True).float().to(self.device)

        elif self.condition == "non-analog":

            if force > self.threshold:

                sound = force*0+torch.rand(1).float().to(self.device) #sound does not depend on force

            else:

                sound = torch.tensor([0.0],requires_grad=True).float().to(self.device)

        sound = torch.tensor([sound]).to(self.device)

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

        return force + (1-torch.rand(1).to(self.device)*2)*self.noise



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

        predictedSound = self.soundPrediction(regulatedForce.to(self.device))



        #we add noise with the noise function

        regulatedForce = self.suckingNoise(regulatedForce)



        regulatedForce2 = self.realBack(regulatedForce.item())

        #print(regulatedForce2,regulatedForce.item())




        self.forces.append(regulatedForce.item())



        #record produced force

        self.producedForcesT.append(regulatedForce2/400)



        #we calculate exhaustion which is how different the sucking force is from the instinct, perhaps this should only work if its higher than instinct 

        exhaustion = self.mseLoss(regulatedForce.to(self.device),instinctForce.float().to(self.device))#abs(regulatedForce-instinctForce)



        #TODO change it so we have baseline as another condition

        #here I want to change the condition based on time to passed to be closer to the original experiement where we have baseline->analog->nonanalog and so on in sequence

        if self.time*10 > self.conditionLength*2: #*2 because first baseline and then the start condition

            if (self.time*10 -self.conditionLength)% self.conditionLength <= 1:

                if self.condition=="analog":

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

            if regulatedForce > self.threshold:

                regulatedForce = regulatedForce*0+torch.rand(1).float().to(self.device) #sound does not depend on force

                self.leftOverSoundCounter = 100

            else:

                if self.leftOverSoundCounter <= 0:

                    regulatedForce = regulatedForce*0#torch.tensor([0.0],requires_grad=True).float().cuda()

                else:
                    regulatedForce = regulatedForce*0+torch.rand(1).float().to(self.device) 

                    self.leftOverSoundCounter -= 1
        #now we know the true sound which is still called regulated force here and find the prediction loss between it and our predicted sound


        predictionLoss = self.mseLoss(predictedSound,torch.tensor(regulatedForce2/400).float().to(self.device))



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




        print(myModel.theX)

        #record results so that we can create the average of the runs 

        averageSound.append(avg)

        averageAmp.append(amp)

        zeros.append(z)

        thresholdHit.append(t)


        #df = pd.DataFrame(myModel.forces)


        #df.to_csv(f"run_{i}_start_{condition}_age_{age}.csv",index=None,header=None)

    return ["averageSound",np.average(averageSound),"averageAmplitude",np.average(averageAmp),"averageZeros",np.average(zeros),"TresholdHits [Analog,nonAnalog]",thresholdHit],freq
epochs = 1

strength = 0.45


memory = 1 #*5 old

learningRate = 0.005
threshold = 0.1667


actuaryNoise = 0.05 


sensoryNoise = 0.05


runs = 5


lossWeights = [1.,1.,1.0] # 0 1 1 young


nonAnalogYoung,fr1  = runExperiment(epochs,"non-analog", actuaryNoise, learningRate,memory,threshold,runs,"young",sensoryNoise,[0.0,1.0,1.0],strength)
analogYoung,fr2 = runExperiment(epochs,"analog", actuaryNoise, learningRate,memory,threshold,runs,"young",sensoryNoise,[0.0,1.0,1.0],strength)

nonAnalogOld,fr3 = runExperiment(epochs,"non-analog", actuaryNoise, learningRate,memory*5,threshold,runs,"old",sensoryNoise,lossWeights,strength)

analogOld,fr4 = runExperiment(epochs,"analog", actuaryNoise, learningRate,memory*5,threshold,runs,"old",sensoryNoise,lossWeights,strength)
#%%
plt.plot(fr1)

plt.show()


plt.plot(fr2)
plt.show()


plt.plot(fr3)
plt.show()
plt.plot(fr4)
plt.show()
