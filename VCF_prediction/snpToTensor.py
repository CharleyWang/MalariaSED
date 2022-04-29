import subprocess
import os
import sys
from operator import itemgetter
import re
import math
import numpy as np
import tensorflow as tf

#assuming all input bed have same length
def fastaGenerate(bedRef, bedAlt,genomeFastaFile):
    #fastaFromBed -fi ../genome/PlasmoDB-26_Pfalciparum3D7_Genome.fasta -bed test.ref.bed -fo aaaa -tab   -name+
    print('generate fasta file for reference sequence')
    subprocess.call(' '.join(['fastaFromBed -fi', genomeFastaFile, '-tab -name+ -bed', bedRef, '-fo test.bed.REF.fasta']), shell = True)
    print('generate fasta file for alternate sequences')
    print('##################################################')
    subprocess.call(' '.join(['fastaFromBed -fi', genomeFastaFile, '-tab -name+ -bed', bedAlt, '-fo test.bed.ALT.fasta']), shell = True)
    with open ('test.bed.ALT.convert.fasta', 'w') as fw:
        with open ('test.bed.ALT.fasta', 'r') as f:
            l = f.readline()
            while l:
                inf, seq = l.strip('\n').split('\t')
                #9161,TCA,*::chr1:9110-9213
                inf2, siteInf = inf.split('::')
                siteInf_s = siteInf.split(':')[1].split('-')[0]
                site, ref, alt = inf2.split(',')
                pickUPsite = int(site) - int(siteInf_s) - 1
                refSeq = seq[pickUPsite:(pickUPsite + len(ref))]
                #if ref != refSeq:
                #    sys.exit('sequence have issue at' + 'inf')
                #else:
                newSeq = seq[:pickUPsite] + alt + seq[(pickUPsite + len(ref)):]
                newSeq = re.sub('\*','', newSeq)
                fw.write(inf + '\t' + newSeq + '\n')
                l = f.readline()

    return(['test.bed.REF.fasta', 'test.bed.ALT.convert.fasta'])

def changeString(string, place, alt):
    string = list(string)
    string[place] = alt
    return(''.join(string))

def snpBedGenerate(snpInputFile, genomeLengthFile, bin_extend = 1000):
    hashG = {}
    with open (genomeLengthFile, 'r') as f:
        ls = f.readlines()
    for l in ls:
        chr, start, end = l.strip('\n').split('\t')
        hashG[chr] = int(end)
    #hashG contains the length of each chromosome
    
    print('Checking the format of VCF file')
    list_ALT = [];list_REF = [];
    with open (snpInputFile, 'r') as f:
        l = f.readline()
        while l:
            chr, site, ref, alts = itemgetter (0,1,2,3) (l.strip('\n').split('\t'))
            if chr.startswith('Pf3D7_'):
                chr = re.sub('Pf3D7_0', 'chr', chr)
                chr = re.sub('Pf3D7_', 'chr', chr)
                chr = re.sub('_v3$', '', chr)
            matchChr = re.search('^chr[1-9]*', chr)
            if not matchChr:
                print('ERROR: the line in the input VCF file with unrecognized chromosome ID \"' + chr + '\".')
                sys.exit()
            if not site.isdigit():
                print('ERROR: the line in the input VCF file with unrecognized string at the second column with position ID \"' + site + '\".')
                sys.exit()
            if [x for x in ref if x not in ['A','T','C','G']]:
                print('ERROR: the line in the input VCF file with unrecognized string at the third column with reference sequence \"' + ref + '\".')
                print('pease make sure reference sequence only contains \'A\',\'T\',\'C\',\'G\'')
                sys.exit()
            if [x for x in alts if x not in ['A','T','C','G',',','*']]:
                print('ERROR: the line in the input VCF file with unrecognized string at the fouth column with alternate  sequence \"' + alts + '\".')
                print('pease make sure reference sequence only contains \'A\',\'T\',\'C\',\'G\',\',\',\'*\'')
                sys.exit()
            

            alts = alts.split(r',')
            for alt in alts:
                lengthExtend = 0
                lALT = len(alt); lREF = len(ref) 		
                if alt == '*':
                    lengthExtend = len(ref)
                else:
                    lengthExtend = lREF - lALT
                
                neededDNA_ref = int((bin_extend)/2)
                neededDNA_alt = (bin_extend + lengthExtend)/2
                leftREF = -neededDNA_ref
                righREF = neededDNA_ref
                    
                if '.5' in str(neededDNA_alt):
                    leftALT = -math.floor(neededDNA_alt)
                    righALT = math.ceil(neededDNA_alt)
                else:
                    leftALT = -int(neededDNA_alt)
                    righALT = int(neededDNA_alt)   
                sALT = int(site) + leftALT; eALT = int(site) + righALT;
                sREF = int(site) + leftREF; eREF = int(site) + righREF;
                if sALT > 0 and eALT <= hashG[chr] and righALT > 0 and sREF > 0 and eREF <= hashG[chr] and righREF > 0:
                    list_ALT.append([chr, str(sALT), str(eALT), site, ref, alt])
                    list_REF.append([chr, str(sREF), str(eREF), site, ref, alt])
                l = f.readline()
    print('VCF format checking finished! Successful!!')
    print('generate bed file')
    with open ('test.alt.bed', 'w') as fw:
        for x in list_ALT:
            fw.write('\t'.join(x[:3]) + '\t' +','.join(x[3:]) + '\n')
    with open ('test.ref.bed', 'w') as fw:
        for x in list_REF:
            fw.write('\t'.join(x[:3]) + '\t' +','.join(x[3:]) + '\n')
    return('test.ref.bed', 'test.alt.bed')
    

def main(*arg):
    snpFILE, genomeLengthFile, genomeFastaFile = arg
    bedFile = snpBedGenerate(snpFILE, genomeLengthFile)
    [refF, testF] = fastaGenerate(bedFile, genomeFastaFile)

if __name__ == "__main__":
    snpF, genomeLenF, genomeFastaF = sys.argv[1:]
    main(snpF, genomeLenF, genomeFastaF)
    #snpBedGenerate(snpF, genomeLengthFile = genomeLenF)




