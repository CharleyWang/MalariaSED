import subprocess
import os
import sys
from operator import itemgetter
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset, TFRecordDataset
from tensorflow.data.experimental import TFRecordWriter

def generateBedGenome(chrLengthFileLoc, binLength = 200, geneFileLoc = 'NA', binExtendLength = 1000):
        binLength = int(binLength)
        binExtendLength = int(binExtendLength)
        binExtendEitherSide = math.floor((binExtendLength - binLength)/2)
        print(binExtendEitherSide)
        with open (chrLengthFileLoc, 'r') as f:
            ls = f.readlines()
        with open ('test.bin.length.genome', 'w') as fw:
            for ele in ls:
                ele = ele.strip('\n')
                [chr, start, end] = ele.split('\t')
                end = int(end)
                numBin = math.floor(end/binLength)
                for i in range(numBin):
                    s = 1 + binLength * i
                    e = binLength * (i+1)
                    if (s - binExtendEitherSide - 1 > 0 and e + binExtendEitherSide < end):
                        fw.write(chr + '\t' + str(s) + '\t' + str(e) + '\n')
        if geneFileLoc != 'NA':
            subprocess.call(' '.join(['intersectBed -a test.bin.length.genome', '-b', geneFileLoc, r'-f 1 -v > test.bed']),shell = True)
            subprocess.call('mv test.bed test.bin.length.genome', shell = True)

def PositiveNegativeBed(peakFile, bedFile):
        subprocess.call(' '.join(['intersectBed -a', bedFile, '-b', peakFile, '-wa -f 0.6 > test.positive']), shell = True)
        subprocess.call(' '.join(['intersectBed -a', bedFile, '-b', peakFile, '-wa -v -f 0.2  > test.negative']), shell = True)

#assuming all input bed have same length
def fastaGenerate(bedFile,binExtend,genomeFastaFile):
        with open ('test.bed.extract.txt', 'w') as fw:
            with open (bedFile, 'r') as f:
                line      = f.readline()
                [s, e]    = itemgetter (1,2) (line.split('\t'))
                lengthBed = int(e) - int(s) + 1
                window    = math.floor((int(binExtend) - lengthBed)/2)
            
                while line:
                    line = line.strip('\n')
                    [chr, s, e]    = itemgetter (0, 1,2) (line.split('\t'))  
                    s = int(s) - window - 1; e = int(e) + window
                    fw.write('\t'.join([chr,str(s),str(e)]) + '\n')
                    line = f.readline()

        subprocess.call(' '.join(['fastaFromBed -fi', genomeFastaFile, '-tab -bed test.bed.extract.txt > test.bed.extract.fasta']), shell = True)


def fastaToCsvFile(fastaTabFile, csvFile = 'aa.csv'):
    hash_ATCG = {'A':[1,0,0,0], 'T':[0,1,0,0], 'C':[0,0,1,0], 'G':[0,0,0,1]}
    dList = []#0010 array
    nList = []#name
    with open (fastaTabFile, 'r') as f:
        l = f.readline()
        while l:
            l = l.strip('\n')
            [ele, seq] = itemgetter (0,1) (l.split('\t'))
            ll_ATCG = []
            for x in seq:
                ll_ATCG.append(hash_ATCG[x])
            dList.append(ll_ATCG)
            nList.append(ele)
            l = f.readline()
    with open (csvFile, 'w') as fw:        
        for x in range(len(dList)):
            fw.write(nList[x] + ',')
            fw.write(','.join([str(y) for z in dList[x] for y in z]))
            fw.write('\n')



def fastaToTensor(fastaTabFile):
    hash_ATCG = {'A':[1,0,0,0], 'T':[0,1,0,0], 'C':[0,0,1,0], 'G':[0,0,0,1]}
    dList = []#0010 array
    nList = []#name
    with open (fastaTabFile, 'r') as f:
        l = f.readline()
        while l:
            l = l.strip('\n')
            [ele, seq] = itemgetter (0,1) (l.split('\t'))
            ll_ATCG = []
            for x in seq:
                ll_ATCG.append(hash_ATCG[x])
            dList.append(ll_ATCG)
            nList.append(ele)
            l = f.readline()
    dList = np.array(dList)
    dList = tf.convert_to_tensor(dList, dtype=tf.uint8)
    return([dList,nList])

def tensorSave(inputTF, saveFile = 'aa'):
    ds = Dataset.from_tensor_slices(inputTF)
    ds_bytes = ds.map(tf.io.serialize_tensor)
    writer = TFRecordWriter(saveFile)
    writer.write(ds_bytes)
    print('successful save file' + saveFile)

def tensorLoad(inputFile = 'aa'):
    ds_bytes = TFRecordDataset(inputFile)
    ds = ds_bytes.map(lambda x: tf.io.parse_tensor(x, out_type=tf.uint8))
    dsNew = tf.stack([x for x in ds], axis=0)
    dsNew = tf.dtypes.cast(dsNew, tf.float32)
    return(dsNew)
    #print(dsNew.shape)
    #x = 14; y = 993; z = 1000
    #print(dsNew[x,y:z])


#def saveTensorByChromosome(inputFastaFile, saveFolder):
#    fasta

def main(*arg):
    [chrLengthFileLoc, fastaFileLoc, peakFile, geneFileLoc, binLength, binExtendLength] = arg
    generateBedGenome(chrLengthFileLoc, binLength, geneFileLoc, binExtendLength)
    PositiveNegativeBed(peakFile, 'test.bin.length.genome')
    fastaGenerate('test.positive', binExtendLength, fastaFileLoc)
    fastaToCsvFile('test.bed.extract.fasta', 'csvData/positive.csv')
    #[tensor_positive, tensor_Name_positive] =  fastaToTensor('test.bed.extract.fasta')
    #tensorSave(tensor_positive, 'tensorData/positive.tensor')
    
    fastaGenerate('test.negative', binExtendLength, fastaFileLoc)
    fastaToCsvFile('test.bed.extract.fasta', 'csvData/negative.csv')
    #[tensor_negative, tensor_Name_negative] =  fastaToTensor('test.bed.extract.fasta')
    #tensorSave(tensor_negative, 'tensorData/negative.tensor')

if __name__ == '__main__':
        [chrLengthFileLocation, fastaFileLocation, peakFileLocation, geneFileLocation, binLengthUsed, binExtendLengthUsed] = sys.argv[1:]
        main(chrLengthFileLocation, fastaFileLocation, peakFileLocation, geneFileLocation, binLengthUsed, binExtendLengthUsed)
        #generateBedGenome(chrLengthFileLoc, binLength, geneFileLoc)
        #PositiveNegativeBed(peakFile, 'test.bin.length.genome')
        #fastaGenerate('test.positive', binExtendLength, fastaFileLoc)
        #tensor_t =  fastaToTensor('test.bed.extract.fasta')
        #tensorSave(tensor_t)
        #tensorLoad('tensorData/negative.tensor')
#python genome_2_tensor.py ../genome/genome_information_pl9.0.bed ../genome/PlasmoDB-26_Pfalciparum3D7_Genome.fasta ../time_specific_peak_bed/mergeData/h_05_10.bed ../genome/PlasmoDB-26_Pfalciparum3D7.gene 200 1000
