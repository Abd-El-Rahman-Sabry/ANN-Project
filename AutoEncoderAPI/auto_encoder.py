import re
from typing import Any
from data_prep import * 
import torch 
from torch.utils.data import Dataset , DataLoader
import torch.nn as nn 
import torch.optim as opt 
import matplotlib.pyplot as plt 
from trainer import Trainer
from sklearn.preprocessing import StandardScaler
from joblib import load , dump


class AutoEncoderDataset(Dataset):


    def __init__(self, dataset : SpeechDataset, number_of_frames) -> None:
        super().__init__()

        self.__ref = dataset 
        self.__number_of_frames = number_of_frames
        self.__uttarance_samples = dataset.training_samples + dataset.testing_samples 
        self.__encoder_input_size = number_of_frames * dataset.count_filters 


        self.__samples = []
        self.__divide_utterance()

        self.__samples = np.concatenate(self.__samples , axis=0)
        print(self.__samples.shape)

        self.__scaler = StandardScaler()
        self.__samples = self.__scaler.fit_transform(self.__samples)
        self.__samples = list(self.__samples)
        dump(self.__scaler , f'AE_SC.bin', compress=True)
        print("The Maximum Value is : " , np.max(self.__samples))
        print("The Minimum Value is : " , np.min(self.__samples))
        self.__samples = [np.array(i).reshape([1 , -1]) for i in self.__samples]

        print(len(self.__samples))
        print(self.__samples[0].shape)

    def __divide_utterance(self):
        
        for sound , label in self.__uttarance_samples:
            utterance_length = sound.shape[1]
            if utterance_length%self.__encoder_input_size != 0 :
                pad_size = self.__encoder_input_size - utterance_length%self.__encoder_input_size
                padding = np.zeros([1 ,pad_size])
                sound = np.concatenate([sound , padding] , axis=1)
            else:
                pad_size = 0

            self.__samples += list(np.array_split(sound , (utterance_length + pad_size)//self.__encoder_input_size , axis=1))

    
    def __len__(self):
        return len(self.__samples)
    
    def __getitem__(self, index) -> Any:
        return torch.from_numpy(self.__samples[index].astype(np.float32))

    @property
    def scaler(self):
        return self.__scaler
    

class AutoEncoder(nn.Module):


    def __init__(self, input_size , code_size , hidden_count , scaler) -> None:
        super().__init__()

        self.__input_size = input_size
        self.__code_size = code_size
        self.__number_of_frames = input_size//code_size
        self.__scaler = scaler

        

        diff = int((input_size - code_size)/hidden_count) - 1

        encoder_hidden = []
        decoder_hidden = []
        for i in range(hidden_count):
            encoder_hidden.append(nn.Linear(input_size - diff*i , input_size - diff*(i+1)))
            encoder_hidden.append(nn.ReLU())

            
            decoder_hidden.append(nn.Linear(input_size - diff*(i+1) , input_size - diff*(i)))
            decoder_hidden.append(nn.ReLU())

        decoder_hidden.reverse()

        print(diff)

        self.__encoder = nn.Sequential(
            *encoder_hidden,
            nn.Linear(input_size - (hidden_count )*diff , code_size),
        
        )

        self.__decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(code_size ,input_size -  hidden_count*diff),
            *decoder_hidden,
        )

    def forward(self,  x):
        x = self.__encoder(x)
        return self.__decoder(x)
    
    def encode(self, val):
        
        input_length = val.shape[1]
        
        if input_length%self.__input_size != 0:
            pad_size = self.__input_size - input_length%self.__input_size
            padding = np.zeros([1 , pad_size])
            val = np.concatenate([val , padding] , axis = 1)
        
        frames = [i.numpy() for i in torch.split(torch.from_numpy(val) ,self.__code_size , 1)]
        
        with torch.no_grad():
            encoded_frame = self.__encoder(torch.from_numpy(self.__scaler.transform(np.concatenate(frames[:self.__number_of_frames] , axis=1)).astype(np.float32))).numpy()

            frames = frames[self.__number_of_frames:]
            
            frames = [frames[i:i+self.__number_of_frames-1] for i in range(0 , len(frames) , self.__number_of_frames)]
            
            for frame in frames:
                encoded_frame = self.__encoder(torch.from_numpy(self.__scaler.transform(np.concatenate([encoded_frame , *frame] , axis=1)).astype(np.float32))).numpy()

            return encoded_frame
    
    
    @property
    def input_size(self):
        return self.__input_size
    
    @property
    def code_size(self):
        return self.__code_size
    
    @classmethod
    def from_pretrained(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Can't find this pretrained file")
        
        matches = re.findall(r'AE_(\d+)_(\d+)_(\d+)' , path)[0]
        params = [int(i) for i in matches]
        scaler = load(f'AE_SC.bin')
        obj = cls(*params, scaler)
        obj.load_state_dict(torch.load(path))

        return obj
        
         


def train_encoder():
    ref = SpeechDataset()

    number_of_frames_within_encoder_input = 2
    
    number_of_hidden_layers = 4
    
    dataset = AutoEncoderDataset(ref , number_of_frames_within_encoder_input)
    
    samples_within_single_frame = ref.count_filters 
    
    encoder_input_size = number_of_frames_within_encoder_input*samples_within_single_frame
    
    encoder = AutoEncoder(encoder_input_size , samples_within_single_frame , number_of_hidden_layers , dataset.scaler)

    
    hyperparams = {"epochs" : 30, "batch_size" : 16 , "lr" : 1e-3}
    encoder_trainer = Trainer(encoder,
                              dataset , 
                              None, 
                              hyperparams,
                              opt.AdamW,
                              nn.MSELoss, 
                              encoder=True,
                              name = f"AE_{encoder_input_size}_{samples_within_single_frame}_{number_of_hidden_layers}f.pt")
    
    encoder_trainer.train()



if __name__ == "__main__":
    train_encoder()
    
        
    
    