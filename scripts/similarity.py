import os
import numpy as np
from astropy.table import Table
from skimage import io
from skimage.util import img_as_float
import multiprocessing

PATH_DATA = '/home/gb/proj/kaggle/data'
PATH_TRAINING_DATA = os.path.join(PATH_DATA, 'images_training_rev1')
PATH_BENCHMARK_DATA = os.path.join(PATH_DATA, 'images_test_rev1')

BENCHMARK_IDS = np.load(os.path.join(PATH_DATA, 'benchmark-zeros.npy'))['galaxyid']
TRAINING_TABLE = Table.read(os.path.join(PATH_DATA, 'training-solutions.fits'))
TRAINING_IDS = TRAINING_TABLE['GalaxyID']
CUTOUT_SIZE = 80
SHRINK = 2


def image2matrix(filename):
    r = CUTOUT_SIZE/2
    width, height = r/SHRINK, r/SHRINK
    img = img_as_float(io.imread(filename))[212-r:212+r, 212-r:212+r, 0]
    return img.reshape(height, img.shape[0]/height,
                       width, img.shape[1]/width).mean(axis=1).mean(axis=2)

def chi2(filename1, filename2):
    return np.mean((image2matrix(filename1) - image2matrix(filename2))**2)

def nearest_neighbour(galaxyid_bm):
    filename_bm = os.path.join(PATH_BENCHMARK_DATA, str(galaxyid_bm)+'.jpg')
    img_bm = image2matrix(filename_bm)
    chi2 = []
    for galaxyid_trn in TRAINING_IDS:
        filename_trn = os.path.join(PATH_TRAINING_DATA, str(galaxyid_trn)+'.jpg')
        img_trn = image2matrix(filename_trn)
        chi2.append( np.mean((img_bm - img_trn)**2) )
    winner_idx = np.argsort(np.array(chi2))[0]
    return (galaxyid_bm, TRAINING_IDS[winner_idx])



if __name__ == '__main__':
    p = multiprocessing.Pool()
    results = p.imap(nearest_neighbour, BENCHMARK_IDS)

    csv = open('neighbours.csv', 'w')
    csv.write('#galid_bm,galid_trn\n')
    for r in results:
        csv.write('{0},{1}\n'.format(*r))
        csv.flush()