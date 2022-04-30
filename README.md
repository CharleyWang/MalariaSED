# MalariaSED
#### MalariaSED is a sequence-based Deep Learning (DL) framework in malaria parasites to understand the contribution of noncoding variants to epigenetic profiles. The current version is able to predict the chromatin impacts, including open chromatin accessibility, H3K9ac, and six TFs, including PfAP2-G, PfAP2-I, PfBDP1, PfAP2-G5, PbAP2-O, and PbAP2-G2, covering different parasite living environments like the mosquito host, the human liver, and human blood cells. 


## Setup
1.bedtools (>2.30.0)

2.tensorflow (2.4.1)

3.numpy (1.19.5)

4.bisect

5.pickle

6.json


## To download and install MalariaSED
#### Clone the repository
`git clone https://github.com/CharleyWang/MalariaSED.git`

`cd MalariaSED`


#### Remove the incompleted files, which are not be directly downloaded by git due to file size. 
`rm TRAINED_model.tar.gz`

`rm intergenic.SNP.dl.logRatio.list.hash.pickle`


#### Download the big trained model file and the pickle file
`wget -c "https://github.com/CharleyWang/MalariaSED/raw/master/TRAINED_model.tar.gz"`

`wget -c "https://github.com/CharleyWang/MalariaSED/raw/master/intergenic.SNP.dl.logRatio.list.hash.pickle"`


#### Decompress the tar.gz file
`tar -xzvf TRAINED_model.tar.gz`

`cd TRAINED_model/`

`gunzip *`


## Running Inference
We provide two input formats for users to compute the chromatin profile changes:

### VCF_prediction
The VCF format requests that the beginning four columns of the user input file include chromosome ID, genomic variant location, reference nucleotide (the nucleotide sequence for insertion or deletion) and alternative nucleotide (‘*for deletion). The nucleotide sequence length should be shorter than 1kb.

To run the prediction model, please use the following command:

`cd VCF_prediction `

`python snpToTensor.DLpredict.py <Variant file> <chromosome length file> <genome fasta file> <output file>`

where variant file should contain four columns including chromosomes, variant sites, reference and alternation sequences. We listed an example of variant file as 'example.variants'. 'chromosome length file' is a bed file listing start and end site of each chromosome, while 'genome fasta file' is the fasta sequence information of P.falciparum genome. 

Here is an exapmle

`cd VCF_prediction `

`python snpToTensor.DLpredict.py example.variants  ../genome/genome_information_pl9.0.bed ../genome/PlasmoDB-26_Pfalciparum3D7_Genome.fasta outputFILE`

#### Output formats from VCF_prediction


### FASTA_prediction
Users can upload two Fasta files for reference and alternation sequences. Multiple sequences are allowed, and MalaraiSED will calculate chromatin effects between two Fasta sequences with the same row ID in the reference and alternation files. The length of both Fasta files should be equal to 1kb.

To run the prediction model, please use the following command:

`cd FASTA_prediction`

`python fastaToTensor.DLpredict.py <Reference_fasta_file>  <Alternation_fasta_file>  <output file>`

where <Reference_fasta_file> indicates a fasta file for reference, while <Alternation_fasta_file> for alternate sequence. The sequence length within both of the files should be 1kb, and have the same.

Here is an exapmle

`cd FASTA_prediction`

`python fastaToTensor.DLpredict.py example.REF.fasta example.ALT.fasta outputFile`
