#
# extract the input matrix from a fasttext bin model into a .npy numpy file (for faster loading )
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.common import  Tqdm
Tqdm.default_mininterval = 1

import fastText
import numpy

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='path to out file', required=True)

parser.add_argument('--fasttext-model', action='store', dest='model',
                    help='.bin model of fasttext', required=True)

args = parser.parse_args()


#
# work
# -------------------------------
# 
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray
    
    
    
model = fastText.load_model(args.model)


in_matrix = model.get_input_matrix()
del model

print("shape:",in_matrix.shape)

z = numpy.zeros((1,in_matrix.shape[1]), dtype=numpy.float32)

in_matrix = numpy.append(z,in_matrix,axis=0)

#in_matrix = bin_ndarray(in_matrix,(in_matrix.shape[0],100))

print("shape (with padding):",in_matrix.shape)

numpy.save(args.out_file,in_matrix)


