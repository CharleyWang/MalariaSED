import genome_2_tensor as g2t
#import snpToTensor as snp2t
import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import sample,seed
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from keras_tuner  import Hyperband
import json
import re
import math
from bisect import bisect_right
import pickle
import os

def modelPredict(dl_model_h5, tensorInputREF, tensorInputALT):
    model = keras.models.load_model(dl_model_h5)
    predPREF = model.predict(tensorInputREF)
    predPALT = model.predict(tensorInputALT)
    predLableREF = model.predict_classes(tensorInputREF)
    predLableALT = model.predict_classes(tensorInputALT)
    #for x in range(len(predP)):
    #    print(predP[x], predLable[x])
    return([predPREF, predLableREF, predPALT, predLableALT])

def logRatio(ref, alt):
    if ref == 1: ref = 1 - 1e-7
    if alt == 1: alt = 1 - 1e-7
    if alt == ref: return(0)
    logRatioREF = np.log2(ref/(1-ref))
    logRatioALT = np.log2(alt/(1-alt))
    return(abs(logRatioREF - logRatioALT))

def E_value_cal(query,ll, lengthLL = 2787184):
	l = ll
	i = bisect_right(l, query)
	return(round((lengthLL - i - 1)/lengthLL, 3))

def value2str(ll):
    return([str(x) for x in ll])

def main(*arg):
    refF, altF, outputFile = arg
    [tensor_Ref, tensor_Name_Ref] =  g2t.fastaToTensor(refF)
    [tensor_Alt, tensor_Name_Alt] =  g2t.fastaToTensor(altF)
    if tensor_Name_Ref == tensor_Name_Alt:
        pass
        print('Fasta file cross checking finished! Successful')
    elif len(tensor_Name_Ref) != len(tensor_Name_Alt):
        print('ERROR: The number of the fasta head, starting with ">", in reference and  alternate sequence files are not the same')
        sys.exit()
    else:
        for i in range(len(tensor_Name_Ref)):
            if tensor_Name_Ref[i] != tensor_Name_Alt[i]:
                print('ERROR: The fasta sequence names in the reference \"' + tensor_Name_Ref[i] + '\" is not same as \"' + tensor_Name_Alt[i]+ '\" in the alternate sequences' )
                sys.exit()
    
    print('Processing the reference Fasta file')
    print('Processing the alternate sequence file')
    #print(tensor_Ref.shape)
    #print(tensor_Alt.shape)
    print('########################################################################')
    with open ('../intergenic.SNP.dl.logRatio.list.hash.pickle', 'rb') as f:
        l_h = pickle.load(f)

    refP1, refL1, altP1, altL1 = modelPredict('../TRAINED_model/atac_05_10h_LSTM.h5', tensor_Ref, tensor_Alt)
    refP2, refL2, altP2, altL2 = modelPredict('../TRAINED_model/atac_15_20h_LSTM.h5', tensor_Ref, tensor_Alt)
    refP3, refL3, altP3, altL3 = modelPredict('../TRAINED_model/atac_25_30h_LSTM.h5', tensor_Ref, tensor_Alt)
    refP4, refL4, altP4, altL4 = modelPredict('../TRAINED_model/atac_35_40h_LSTM.h5', tensor_Ref, tensor_Alt)
    refP5, refL5, altP5, altL5 = modelPredict('../TRAINED_model/pfBDP_troph.h5', tensor_Ref, tensor_Alt)
    refP6, refL6, altP6, altL6 = modelPredict('../TRAINED_model/pfBDP_schizont.h5', tensor_Ref, tensor_Alt)
    refP7, refL7, altP7, altL7 = modelPredict('../TRAINED_model/ap2G.schizont.h5', tensor_Ref, tensor_Alt)
    refP8, refL8, altP8, altL8 = modelPredict('../TRAINED_model/ap2G.sex_ring.h5', tensor_Ref, tensor_Alt)
    refP9, refL9, altP9, altL9 = modelPredict('../TRAINED_model/ap2G.gam.h5', tensor_Ref, tensor_Alt)
    refP10, refL10, altP10, altL10 = modelPredict('../TRAINED_model/AP2I_schizont.h5', tensor_Ref, tensor_Alt)
    refP11, refL11, altP11, altL11 = modelPredict('../TRAINED_model/H3K9ac_troph.h5', tensor_Ref, tensor_Alt)
    refP12, refL12, altP12, altL12 = modelPredict('../TRAINED_model/H3K9ac_schizont.h5', tensor_Ref, tensor_Alt)
    refP13, refL13, altP13, altL13 = modelPredict('../TRAINED_model/AP2G5_troph.h5', tensor_Ref, tensor_Alt)
    
    with open (outputFile, 'w') as fw:
        fw.write('\t'.join(value2str(['site,ref,alt::regionSelected', 'REF_Prob_OpenChromatin_Ring',\
                  'ALT_Prob_OpenChromatin_Ring',\
                  'REF_Prob_OpenChromatin_EarlyTroph', 'ALT_Prob_OpenChromatin_EearlyTroph',\
                  'REF_Prob_OpenChromatin_LateTroph', 'ALT_Prob_OpenChromatin_LateTroph',\
                  'REF_Prob_OpenChromatin_Schizont', 'ALT_Prob_OpenChromatin_Schizont',\
                  'REF_Prob_pfBDP_Troph', 'ALT_Prob_pfBDP_Troph',\
                  'REF_Prob_pfBDP_Schizont', 'ALT_Prob_pfBDP_Schizont',\
                  'REF_Prob_Ap2G_Schizont', 'ALT_Prob_Ap2G_Schizont',\
                  'REF_Prob_Ap2G_Sex_ring', 'ALT_Prob_Ap2G_Sex_ring',\
                  'REF_Prob_Ap2G_Gametocyte', 'ALT_Prob_Ap2G_Gametocyte',\
                  'REF_Prob_Ap2I_Schizont', 'ALT_Prob_Ap2I_Schizont',\
                  'REF_Prob_H3K9ac_Troph', 'ALT_Prob_H3K9ac_Troph',\
                  'REF_Prob_H3K9ac_Schizont', 'ALT_Prob_H3K9ac_Schizont',\
                  'REF_Prob_Ap2G5_Troph', 'ALT_Prob_Ap2G5_Troph',\
                  'logRatio_OpenChromatin_Ring', 'logRatio_OpenChromatin_EarlyTroph',\
                  'logRatio_OpenChromatin_LateTroph', 'logRatio_OpenChromatin_Schizont',\
                  'logRatio_pfBDP_Troph', 'logRatio_pfBDP_Schizont',\
                  'logRatio_Ap2G_Schizont', 'logRatio_Ap2G_Sex_ring',\
                  'logRatio_Ap2G_Sex_ring', 'logRatio_Ap2I_Schizont',\
                  'logRatio_H3K9ac_Troph', 'logRatio_H3K9ac_Schizont',\
                  'logRatio_Ap2G5_Troph',\
                  'Eval_OpenChromatin_Ring', 'Eval_OpenChromatin_EarlyTroph',\
                  'Eval_OpenChromatin_LateTroph', 'Eval_OpenChromatin_Schizont',\
                  'Eval_pfBDP_Troph', 'Eval_pfBDP_Schizont',\
                  'Eval_Ap2G_Schizont', 'Eval_Ap2G_Sex_ring',\
                  'Eval_Ap2G_Sex_ring', 'Eval_Ap2I_Schizont',\
                  'Eval_H3K9ac_Troph', 'Eval_H3K9ac_Schizont',\
                  'Eval_Ap2G5_Troph', '\n'])))

        for z in range(len(altL1)):
            fw.write('\t'.join(value2str([tensor_Name_Ref[z], refP1[z][0], altP1[z][0],\
                                  refP2[z][0], altP2[z][0],\
                                  refP3[z][0], altP3[z][0],\
                                  refP4[z][0], altP4[z][0],\
                                  refP5[z][0], altP5[z][0],\
                                  refP6[z][0], altP6[z][0],\
                                  refP7[z][0], altP7[z][0],\
                                  refP8[z][0], altP8[z][0],\
                                  refP9[z][0], altP9[z][0],\
                                  refP10[z][0], altP10[z][0],\
                                  refP11[z][0], altP11[z][0],\
                                  refP12[z][0], altP12[z][0],\
                                  refP13[z][0], altP13[z][0],\
                                  logRatio(refP1[z][0], altP1[z][0]), logRatio(refP2[z][0], altP2[z][0]),\
                                  logRatio(refP3[z][0], altP3[z][0]), logRatio(refP4[z][0], altP4[z][0]),\
                                  logRatio(refP5[z][0], altP5[z][0]), logRatio(refP6[z][0], altP6[z][0]),\
                                  logRatio(refP7[z][0], altP7[z][0]), logRatio(refP8[z][0], altP8[z][0]),\
                                  logRatio(refP9[z][0], altP9[z][0]), logRatio(refP10[z][0], altP10[z][0]),\
                                  logRatio(refP11[z][0], altP11[z][0]), logRatio(refP12[z][0], altP12[z][0]),\
                                  logRatio(refP13[z][0], altP13[z][0]),\
                                  E_value_cal(logRatio(refP1[z][0], altP1[z][0]), l_h['atac1']),\
                                  E_value_cal(logRatio(refP2[z][0], altP2[z][0]), l_h['atac2']),\
                                  E_value_cal(logRatio(refP3[z][0], altP3[z][0]), l_h['atac3']),\
                                  E_value_cal(logRatio(refP4[z][0], altP4[z][0]), l_h['atac4']),\
                                  E_value_cal(logRatio(refP5[z][0], altP5[z][0]), l_h['bdpT']),\
                                  E_value_cal(logRatio(refP6[z][0], altP6[z][0]), l_h['bdpS']),\
                                  E_value_cal(logRatio(refP7[z][0], altP7[z][0]), l_h['ap2G_S']),\
                                  E_value_cal(logRatio(refP8[z][0], altP8[z][0]), l_h['ap2G_SR']),\
                                  E_value_cal(logRatio(refP9[z][0], altP9[z][0]), l_h['ap2G_G']),\
                                  E_value_cal(logRatio(refP10[z][0], altP10[z][0]), l_h['ap2I_S']),\
                                  E_value_cal(logRatio(refP11[z][0], altP11[z][0]), l_h['h3k9ac_T']),\
                                  E_value_cal(logRatio(refP12[z][0], altP12[z][0]), l_h['h3k9ac_S']),\
                                  E_value_cal(logRatio(refP13[z][0], altP13[z][0]), l_h['ap2G5_S']),\
                                   '\n']))
                                  )
    print('Successfully processed all sequences!')
    #os.remove('test.alt.bed')
    #os.remove('test.ref.bed')
    #os.remove('test.bed.ALT.fasta')
    #os.remove('test.bed.REF.fasta')
    #os.remove('test.bed.ALT.convert.fasta')
#['atac1', 'atac2', 'atac3', 'atac4', 'bdpT', 'bdpS', 'ap2G_S', 'ap2G_SR', 'ap2G_G', 'ap2I_S', 'h3k9ac_T', 'h3k9ac_S', 'ap2G5_S']



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ##silence tensorflow
    fastaREF,fastaALT, outputF = sys.argv[1:]
    main(fastaREF, fastaALT, outputF)
    #snpBedGenerate(snpF, genomeLengthFile = genomeLenF)

