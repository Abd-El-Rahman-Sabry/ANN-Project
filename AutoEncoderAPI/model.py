from typing import Any
from data_prep import * 
import torch 
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn 
import torch.optim as opt 
from trainer import Trainer
import matplotlib.pyplot as plt 
from auto_encoder import AutoEncoder

class AvarageFrameTransform:

    def __init__(self , frame_size) -> None:
        self.__frame_size = frame_size

    def __call__(self, sample) -> Any:
        sound , label = sample

        sound = sound.reshape([-1 , self.__frame_size])
        sound = sound.mean(axis=0)

        return torch.from_numpy(sound) , torch.tensor(label)
    
class AutoEncoderTransform:

    def __init__(self , encoder) -> None:
        self.__encoder = encoder

    def __call__(self, sample) -> Any:
        sound , label = sample 
        sound = self.__encoder.encode(sound)

        return torch.from_numpy(sound).squeeze(0), torch.tensor(label)



class DigitSpeechDataset(Dataset):


    def __init__(self, ref , **kwargs) -> None:
        super().__init__()

        self.__ref : SpeechDataset = ref 
        self.__transform = kwargs.get("transform" , None)
        
        is_train = kwargs.get("train" , False)
        is_test = kwargs.get("test" , False)

        if is_train or (not is_test):
            self.__samples = self.__ref.training_samples
        else:
            self.__samples = self.__ref.testing_samples


    def __len__(self):
        return len(self.__samples)
    
    def __getitem__(self, index) -> tuple:
        sample = self.__samples[index]

        if self.__transform:
            sample = self.__transform(sample)

        sound , label = sample 

        return sound , label
    
class DigitClassifier(nn.Module):

    def __init__(self , input_size , hidden_layers , number_of_hidden_layers) -> None:
        super().__init__()
        
        deep_hidden_layers = [nn.Linear(hidden_layers , hidden_layers) , nn.ReLU()]*number_of_hidden_layers
        self.__model = nn.Sequential(
                nn.Linear(input_size , hidden_layers),
                nn.ReLU(),
                *deep_hidden_layers,
                nn.Linear(hidden_layers , 10)
        )

    def forward(self , x):
        return self.__model(x)
    



                
def test_train_avarage(data_ref):

    hyperparams = {"epochs" : 12, "batch_size" : 8 , "lr" : 1e-3}
    
    frame_size = data_ref.count_filters
    train_dataset = DigitSpeechDataset(data_ref , transform = AvarageFrameTransform(frame_size) , train=True)
    test_dataset = DigitSpeechDataset(data_ref , transform = AvarageFrameTransform(frame_size) , train=False)

    model = DigitClassifier(frame_size , 100 , 4)
    
    trainer = Trainer(model, 
                      train_dataset, 
                      test_dataset, 
                      hyperparams , 
                      opt.Adam , 
                      nn.CrossEntropyLoss,
                      name = "Avg_Digit_Classifer.pt")
    trainer.train()
    print(trainer.test_classifier())

def test_train_auto_encoder(data_ref):

    hyperparams = {"epochs" : 10 , "batch_size" : 8 , "lr" : 1e-3}
    
    frame_size = data_ref.count_filters
    encoder = AutoEncoder.from_pretrained("last_states/AE_26_13_4f.pt")

    train_dataset = DigitSpeechDataset(data_ref , transform = AutoEncoderTransform(encoder) , train=True)
    test_dataset = DigitSpeechDataset(data_ref , transform = AutoEncoderTransform(encoder) , train=False)

    model = DigitClassifier(frame_size , 100 , 4)
    
    trainer = Trainer(model, 
                      train_dataset, 
                      test_dataset, 
                      hyperparams , 
                      opt.Adam , 
                      nn.CrossEntropyLoss,
                      name = "AE_Digit_Classifer.pt")
    trainer.train()
    print(trainer.test_classifier())

    

if __name__ == "__main__":
    data_ref = SpeechDataset("SpeechDataset")
    test_train_auto_encoder(data_ref)

