import librosa
import numpy as np 
import os 
import glob 
import enum

import torch


class SpeechDataset:

    def __init__(self , path="SpeechDataset" , frame_lenght = 20 , max_length= None , **kwargs) -> None:
        
        if not os.path.exists(path):
            raise FileNotFoundError("Wrong Path")
        
        self.__path = path 
        self.__trian_path = os.path.join(path , "Train")
        self.__test_path = os.path.join(path , "Test")

        self.__frame_length = frame_lenght
        self.__max_length = 0 if max_length is None else max_length

        self.__number_of_filters = kwargs.get("n_mfcc" , 13)

        self.__sr = 0
        self.__means = []
        self.__divs = []
        self.__train_samples = self.__load_sound_samples(self.__trian_path , True)
        self.__test_samples = self.__load_sound_samples(self.__test_path)

        self.__mean = sum(self.__means)/len(self.__means)
        self.__std = sum(self.__divs)/len(self.__divs)
        
        print("SpeechDataset: Number of training Samples : " , len(self.__train_samples))
        print("SpeechDataset: Number of testing Samples : " ,len(self.__test_samples))
        # Do processing on the sound samples 
        self.__process_sound()


    def __load_sound_samples(self, path ,calc_norm = False):
        samples = []
        for i in range(10):
            wav_files = glob.glob(os.path.join(path , f"*_{i}.wav"))
            for wav in wav_files:
                y , self.__sr = librosa.load(wav)
                # Calculate the Duration 
                self.__max_length = max(self.__max_length , len(y)/self.__sr)
                S = librosa.feature.melspectrogram(y=y, sr=self.__sr, n_mels=self.__number_of_filters,
                                   win_length=self.get_number_of_samples_in_a_frame() ,)
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S)).T
                
                y_samples = mfcc.reshape([1 , -1])

                if calc_norm:
                    meu = np.mean(y_samples)
                    std = np.sqrt((y_samples - meu)**2)
                    self.__means.append(meu)
                    self.__divs.append(std)
                samples.append([y_samples , i])

        return samples
    
    def get_number_of_samples_in_a_frame(self):
        number_of_samples = int(np.ceil(self.__frame_length/1000.0 * self.__sr))
        return number_of_samples
    
    def __process_sound(self,):
        number_of_samples = self.count_filters
        
        # Process Training Dataset 
        post_processing_training = []
        post_processing_testing = []
        
        # Training Loop 
        for sound , label in self.__train_samples:            
            # Zero Padding 
            if sound.shape[1]%number_of_samples != 0:
                zero_padding = np.zeros([1 , number_of_samples - (sound.shape[1]%number_of_samples)])
                sound = np.concatenate([sound , zero_padding] , axis=1)
                # sound = (sound - self.__mean)/self.__std
            
            # sound = sound.reshape([-1 , number_of_samples])

            post_processing_training.append([sound , label])


        # Testing Loop 
        for sound , label in self.__test_samples:            
            # Zero Padding 
            if sound.shape[1]%number_of_samples != 0:
                zero_padding = np.zeros([1 , number_of_samples - (sound.shape[1]%number_of_samples)])
                sound = np.concatenate([sound , zero_padding] , axis=1)
                # sound = ( self.__sound - self.__mean )/self.__std
            
            # sound = sound.reshape([-1 , number_of_samples])
            
            post_processing_testing.append([sound , label])


        self.__train_samples = post_processing_training
        self.__test_samples = post_processing_testing

    @property
    def sr(self):
        return self.__sr
    
    @property
    def frame_size(self):
        return self.__frame_length
    
    @property
    def training_samples(self ,):
        return self.__train_samples
        
    @property
    def testing_samples(self ,):
        return self.__test_samples
    
    @property
    def count_filters(self):
        return self.__number_of_filters
            

            
    




def main():
    print("This main")
    y , sr = librosa.load('C03_3.wav')

    frame_length = 20 # in ms 

    number_of_samples = int(np.ceil(frame_length/1000.0 * sr ))

    y_samples = np.expand_dims(np.array(y) , axis=0)

    print(y_samples.shape)
    zero_padding = np.zeros([1 , number_of_samples - (y_samples.shape[1]%number_of_samples)])
    y_samples = np.concatenate([y_samples , zero_padding] , axis=1)
    print(y_samples.shape)
    y_samples = y_samples - np.mean(y_samples)
    y_samples = y_samples.reshape([-1,number_of_samples])

    
    print(y_samples.shape)

    
    
def test():
    
    sound , sr = librosa.load('C03_3.wav')
    print(len(sound))
    mfcc = librosa.feature.mfcc(y=sound , sr=sr , n_mfcc=128, hop_length=441)
    mfcc = mfcc.T
    mfcc = mfcc.reshape([1 , -1])
    print(mfcc.shape)



if __name__ == "__main__":
    test()

