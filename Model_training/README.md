# MalariaSED_training
#### MalariaSED provides two ways to train the model by using either random splitting or chromosome splitting strategy. We also provide a training file for three-layer convolutional network. We set the kernel number of each convolution layer the same as the DL network previously published in the human genome (PMCID: PMC4768299). 

1. z_dl_random_chr_split: training by chromosome splitting strategy
2. z_dl_random_split: training by random splitting strategy
3. z_dl_random_chr_split: three-layer convolutional network training

You need to download the datasets for 15 chromatin profiles from the following link. 
https://usf.box.com/s/278l0z6qr33res04oasn4wzekct0vqkt

The file 'tl_train_LSTM_bayes.py' in each folder corresponds to the hyperparameter optimization process. You need to change the input file after 'def main():



## Setup
1.bedtools (>2.30.0)

2.tensorflow (2.4.1)

3.numpy (1.19.5)

4.bisect

5.pickle

6.json

7.keras-tuner(1.1.2)
