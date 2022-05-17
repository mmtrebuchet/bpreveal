import tensorflow as tf
def setMemoryGrowth():
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth enabled.")
    except:
        print("Not using GPU")
        pass

def loadChromSizes(fname):
    #Read in a chrom sizes file and return a dictionary mapping chromosome name → size
    ret = dict()
    with open(fname, 'r') as fp:
        for line in fp:
            if(len(line) > 2):
                chrom, size = line.split()
                ret[chrom] = int(size)
    return ret

