from typing import Any
from data_prep import * 
import torch 
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn 
import torch.optim as opt 
import matplotlib.pyplot as plt 
import json 
from datetime import datetime 

class Trainer:


    def __init__(self, model ,train_dataset, test_dataset, hyperparams : dict, optimizer , loss , **kwargs ) -> None:
        self.__model = model 
        self.__train_dataset =train_dataset
        self.__test_dataset = test_dataset 
        self.__hyperparams = hyperparams

        self.__train_encoder = kwargs.get("encoder" , False)

        self.__epochs = hyperparams.get("epochs" , 1)
        self.__batch_size = hyperparams.get("batch_size" , 1)
        self.__lr = hyperparams.get("lr" , 1e-3)
        self.__alr = hyperparams.get("alr", opt.lr_scheduler.LinearLR)


        self.__optimizer = optimizer(self.__model.parameters() , lr=self.__lr)
        self.__scheduler = self.__alr(self.__optimizer)
        self.__loss = loss()    

        self.__device = "cuda" if torch.cuda.is_available() else "cpu"

        self.__name = kwargs.get("name" , "training_state")

        self.__model.to(self.__device)

        self.__last_training_losses = None

    def train(self):

        loader = DataLoader(self.__train_dataset , batch_size=self.__batch_size , shuffle=True)
        total = self.__epochs * len(self.__train_dataset)
        print("The Number of sampels is : " , len(self.__train_dataset))
        losses = []
        counter = 0
        report = ''
        for epoch in range(self.__epochs):
            report += "\nChecking Loss......\n"
            if self.__train_encoder:
                l = self.test_encoder()
                report+= f"Total epoch {epoch} [{round(l , 5)}]"
                losses.append(l)
            for idx , sample in enumerate(loader):
                os.system('cls')
                progress = round(100*counter/total , 4)
                print(f"{report}\n\nCurrent Epoch {epoch} Training Progress ---- > " , progress if progress < 100 else 100 , "%")
                counter += self.__batch_size
                self.__optimizer.zero_grad()
                
                if not self.__train_encoder:
                    sound , label = sample 
                    sound = sound.to(self.__device)
                    label = label.to(self.__device)
                    logits = self.__model(sound)
                    L = self.__loss(logits , label)
                else:
                    sample = sample.to(self.__device)
                    logits = self.__model(sample)
                    L = self.__loss(logits , sample) 

                L.backward()
                self.__optimizer.step()
                #self.__scheduler.step()
                    


                
        print("Training Done !! ----- > 100%")
        self.__last_training_losses = losses
        
        if self.__train_encoder:
            current_acc = self.test_encoder()
        else:
            current_acc = self.test_classifier() 

        states = self.load_states()

        if self.__name in states.keys():
            last_acc = states[self.__name]['acc']
            if not self.__train_encoder:
                if last_acc < current_acc:
                    torch.save(self.__model.state_dict() , fr"last_states\{self.__name}")
                    states[self.__name] = self.__make_state(current_acc)   
            else:
                if last_acc > current_acc:
                    torch.save(self.__model.state_dict() , fr"last_states\{self.__name}")
                    states[self.__name] = self.__make_state(current_acc)   


        else:
            torch.save(self.__model.state_dict() , fr"last_states\{self.__name}")
            states[self.__name] = self.__make_state(current_acc)

        self.save_states(states)

    def test_encoder(self):
        loader = DataLoader(self.__train_dataset , batch_size=self.__batch_size)

        total = len(self.__train_dataset)

        loss_sum = 0
        with torch.no_grad():
            for sample in loader:
                sample = sample.to(self.__device)
                encoded = self.__model(sample)
                error = (sample - encoded)**2
                loss_sum += torch.sum(error).item()

        return loss_sum/total 
                

    def test_classifier(self):

        n_correct = 0
        n_total = len(self.__test_dataset)

        loader = DataLoader(self.__test_dataset , batch_size=self.__batch_size)
        with torch.no_grad():
            for sound , label in loader:
                sound = sound.to(self.__device)
                label = label.to(self.__device)
                out = self.__model(sound)
                _ , predicted = torch.max(out , 1)
                n_correct += (predicted == label).sum().item()
            acc = round(100 * n_correct / n_total , 4)
            print("The Accuarcy is ---->" , acc)
            return acc
        
    def plot_loss(self):
        if self.__last_training_losses:
            plt.plot(self.__last_training_losses)

        else : 
            print("There is no last training !!")


    def load_states(self):
        if not os.path.exists("last_training_states.json"):
            return {}
        
        else: 
            with open("last_training_states.json" , 'r') as f:
                txt = f.read()
                return json.loads(txt)
            
    def save_states(self , states):
        with open("last_training_states.json" , 'w') as f:
            txt = json.dumps(states , indent=4)
            f.write(txt)

    def __make_state(self, acc):
        now = datetime.now()
        t = now.strftime("%H:%M:%S")
        d = now.strftime("%d/%m/%Y")
        return {"acc" : acc , "time" : t , "date" : d}