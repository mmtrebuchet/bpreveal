
def loadChromSizes(fname):
    #Read in a chrom sizes file and return a dictionary mapping chromosome name → size
    ret = dict()
    with open(fname, 'r') as fp:
        for line in fp:
            if(len(line) > 2):
                chrom, size = line.split()
                ret[chrom] = int(size)
    return ret

