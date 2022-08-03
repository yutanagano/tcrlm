'''
Simple implementations of a CDR3-encoding machine which uses atchley factors
(Atchley et al. 2005) to embed each individual amino acid in a CDR3 sequence,
then averages all values in each dimension to produce a fixed-sized vector
embedding for any CDR3 sequence. This is one of the baselines to which the CDR3
BERT model can be compared.
'''


import numpy as np


_atchley_dict = {
    'A': [-0.591,-1.302,-0.733,1.570,-0.146],
    'C': [-1.343,0.465,-0.862,-1.020,-0.255],
    'D': [1.050,0.302,-3.656,-0.259,-3.242],
    'E': [1.357,-1.453,1.477,0.113,-0.837],
    'F': [-1.006,-0.590,1.891,-0.397,0.412],
    'G': [-0.384,1.652,1.330,1.045,2.064],
    'H': [0.336,-0.417,-1.673,-1.474,-0.078],
    'I': [-1.239,-0.547,2.131,0.393,0.816],
    'K': [1.831,-0.561,0.533,-0.277,1.648],
    'L': [-1.019,-0.987,-1.505,1.266,-0.912],
    'M': [-0.663,-1.524,2.219,-1.005,1.212],
    'N': [0.945,0.828,1.299,-0.169,0.933],
    'P': [0.189,2.081,-1.628,0.421,-1.392],
    'Q': [0.931,-0.179,-3.005,-0.503,-1.853],
    'R': [1.538,-0.055,1.502,0.440,2.897],
    'S': [-0.228,1.399,-4.760,0.670,-2.647],
    'T': [-0.032,0.326,2.213,0.908,1.313],
    'V': [-1.337,-0.279,-0.544,1.242,-1.262],
    'W': [-0.595,0.009,0.672,-2.128,-0.184],
    'Y': [0.260,0.830,3.097,-0.838,1.512]
}



def atchley_encode(aa_seq: str) -> np.ndarray:
    embeddings = []
    for aa in aa_seq: embeddings.append(np.array(_atchley_dict[aa]))

    embeddings = np.stack(embeddings)
    averaged = np.mean(embeddings,axis=0)

    return averaged