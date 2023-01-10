# MalariaSED_training
#### MalariaSED provides two ways to train the model by using either random splitting or chromosome splitting strategy. We also provide a training file for three-layer convolutional network. We set the kernel number of each convolution layer the same as the DL network previously published in the human genome (PMCID: PMC4768299). 

1. z_dl_chr_split: training by chromosome splitting strategy
2. z_dl_random_split: training by random splitting strategy
3. z_dl_random_conv: three-layer convolutional network training

Please download the datasets for 15 chromatin profiles from the following link. 
https://usf.box.com/s/278l0z6qr33res04oasn4wzekct0vqkt

The file 'tl_train_LSTM_bayes.py' in each folder corresponds to the hyperparameter optimization process. The file 'genome_2_tensor.py' contains many important functions for successfully running 'tl_train_LSTM_bayes.py'. Please keep the two files together in each folder. You need to change the input file in the beginning four lines of the main function:

`#example for atac_seq for 15-20h data`

`trainAllpos = tensorLoad('time_specific_peak_generateTensor/tensorData_h15_20/positive.tensor_1k')`

`trainAllneg = tensorLoad('time_specific_peak_generateTensor/tensorData_h15_20/negative.tensor_1k')`

`(tfXtrainPos, tfXvalidPos, tfXtestPos) = train_testID_returnSampling('time_specific_peak_generateTensor/tensorData_h15_20/locName.positive_1k.txt', trainAllpos)`

`(tfXtrainNeg, tfXvalidNeg, tfXtestNeg) = train_testID_returnSampling('time_specific_peak_generateTensor/tensorData_h15_20/locName.negative_1k.txt', trainAllneg)`

To run it, just simplely type `python tl_train_LSTM_bayes.py `


## Setup
1.bedtools (>2.30.0)

2.tensorflow (2.4.1)

3.numpy (1.19.5)

4.bisect

5.pickle

6.json

7.keras-tuner(1.1.2)
