#!/usr/bin/env python3
import utils
import json
import h5py
import logging
import numpy as np

def revcomp(oneHotSeq):
    #Since the one-hot encoding is A,C,G,T, flipping the array upside-down complements the sequence,
    #and of course flipping it forward-backwards reverses it.
    return np.flip(oneHotSeq)


def addRevcompSeq(inH5, outH5):
    #Reads in the sequence records from the input hdf5 file, and adds them and their reverse complement to the output. 

    inSeq = np.array(inH5["sequence"])
    numInSeqs = inSeq.shape[0]
    outSeq = np.zeros((numInSeqs*2, inSeq.shape[1], 4))
    for seqid in range(numInSeqs):
        outSeq[seqid*2] = inSeq[seqid]
        outSeq[seqid*2+1] = revcomp(inSeq[seqid])
    outH5.create_dataset("sequence", data=outSeq, dtype='i1', compression='gzip')

def flipDats(inDats, order):
    ret = np.zeros(inDats.shape)
    for outCol, inCol in enumerate(order):
        ret[:,outCol] = np.flip(inDats[:,inCol])
    return ret

def addHeadData(inH5, outH5, headConfig):
    inHead = headConfig["head-id"]
    inDats = np.array(inH5["head_{0:d}".format(inHead)])

    numInDats = inDats.shape[0]

    outDats = np.zeros((numInDats*2, inDats.shape[1], inDats.shape[2]))
    for datid in range(numInDats):
        outDats[datid*2] = inDats[datid]
        outDats[datid*2+1] = flipDats(inDats[datid], headConfig["task-order"])
    outH5.create_dataset("head_{0:d}".format(inHead), data=outDats, dtype='f4', compression='gzip')



def main(config):
    inH5 = h5py.File(config["input-h5"], "r")
    outH5 = h5py.File(config["output-h5"], "w")
    logging.info("Adding reverse-complemented sequence")
    addRevcompSeq(inH5, outH5)

    for head in config["heads"]:
        logging.info("Adding reverse-complemented data for head {0:d}".format(head["head-id"]))
        addHeadData(inH5, outH5, head)
    logging.info("Closing data files.")
    inH5.close()
    outH5.close()






if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "r") as configFp:
        config = json.load(configFp)
    utils.setVerbosity(config["verbosity"])
    main(config)

