"""Plots features."""
import os
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import multiprocessing

from astropy.table import Table, join
from astropy import log

from skimage.util import img_as_float
from skimage import io

# Where is the competition data?
PATH_DATA = '/home/gb/proj/kaggle/data'
PATH_TRAINING_DATA = os.path.join(PATH_DATA, 'images_training_rev1')
PATH_BENCHMARK_DATA = os.path.join(PATH_DATA, 'images_test_rev1')
PATH_TEMPLATES = os.path.join(PATH_DATA, 'templates')
CLASSES = ['Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2',
           'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1',
           'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2',
           'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2',
           'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7',
           'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2',
           'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4',
           'Class11.5', 'Class11.6']

def get_img(path, GalaxyID):
    """Returns the image data of a galaxy as a numpy array."""
    filename = os.path.join(path, '{0}.jpg'.format(GalaxyID))
    return img_as_float(io.imread(filename))  

def shrink(img, width, height):
    """Reduces the resolution of an image to a given width/height."""
    return img.reshape(height, img.shape[0]/height, width, img.shape[1]/width).mean(axis=1).mean(axis=2)

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                     -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    #return height, x, y, width_x, width_y
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p, (errorfunction(p)**2).mean()


class Galaxy(object):

    def __init__(self, GalaxyID, cutout_sizes=[10, 40, 100]):
        self.GalaxyID = GalaxyID
        try:
            self.img = get_img(PATH_TRAINING_DATA, GalaxyID)
        except IOError:
            self.img = get_img(PATH_BENCHMARK_DATA, GalaxyID)
        self.img_grey = (self.img[:, :, 0] + self.img[:, :, 1]
                         + self.img[:, :, 2]) / 3.
        self.cutout_sizes = cutout_sizes
        self.cutouts = [self.cut(cutout_sizes[0]),
                        self.cut(cutout_sizes[1]),
                        self.cut(cutout_sizes[2])]
        self.cutouts_grey = [self.cut(cutout_sizes[0], grey=True),
                             self.cut(cutout_sizes[1], grey=True),
                             self.cut(cutout_sizes[2], grey=True)]

    def cut(self, size, grey=False):
        """Returns a central square cut-out image."""
        r = int(size / 2)
        if grey:
            return self.img_grey[212-r:212+r, 212-r:212+r]
        else:
            return self.img[212-r:212+r, 212-r:212+r]      

    def hsym(self, cutid=0):
        """Computes a metric for the horizontal symmetry."""
        height = self.cutout_sizes[cutid]
        width = int(height/2)
        left_side = self.cutouts_grey[cutid][:,:width]
        right_side = self.cutouts_grey[cutid][:,width:]
        sample1 = shrink(left_side, int(width/5), int(height/5))
        sample2 = shrink(np.fliplr(right_side), int(width/5), int(height/5))
        return np.mean((sample1 - sample2)**2)

    def vsym(self, cutid=0):
        """Computes a metric for the vertical symmetry."""
        width = self.cutout_sizes[cutid]
        height = int(width/2)
        top_side = self.cutouts_grey[cutid][height:,:]
        bottom_side = self.cutouts_grey[cutid][:height,:]
        sample1 = shrink(top_side, int(width/5), int(height/5))
        sample2 = shrink(np.flipud(bottom_side), int(width/5), int(height/5))
        return np.mean((sample1 - sample2)**2)

    def std(self, cutid=0): return np.std(self.cutouts_grey[cutid] / self.cutouts_grey[cutid].mean())

    def red(self, cutid=0): return self.cutouts[cutid][:, :, 0].mean()

    def green(self, cutid=0): return self.cutouts[cutid][:, :, 1].mean()

    def blue(self, cutid=0): return self.cutouts[cutid][:, :, 2].mean()

    def gaussfit(self, cutid=2):
        fit, goodness = fitgaussian(self.cutouts_grey[cutid])
        width_x, width_y = fit[3], fit[4]
        return width_x, width_y, goodness

    def features(self):
        """Returns a dictionary containing all the computed features."""
        feat = {}
        for name in ['GalaxyID']:
            feat[name] = self.__getattribute__(name)

        for cutid, size in enumerate(self.cutout_sizes):
            feat['min_{0}'.format(size)] = self.cutouts_grey[cutid].min()
            feat['max_{0}'.format(size)] = self.cutouts_grey[cutid].max()
            feat['std_{0}'.format(size)] = self.cutouts_grey[cutid].std()
            feat['std2_{0}'.format(size)] = np.std(self.cutouts_grey[cutid] / self.cutouts_grey[cutid].mean())
            feat['p10_{0}'.format(size)] = np.percentile(self.cutouts_grey[cutid], 10)
            feat['p50_{0}'.format(size)] = np.percentile(self.cutouts_grey[cutid], 50)
            feat['p90_{0}'.format(size)] = np.percentile(self.cutouts_grey[cutid], 90)
            feat['red_{0}'.format(size)] = self.red(cutid)
            feat['green_{0}'.format(size)] = self.green(cutid)
            feat['blue_{0}'.format(size)] = self.blue(cutid)
            feat['hsym_{0}'.format(size)] = self.hsym(cutid)
            feat['vsym_{0}'.format(size)] = self.vsym(cutid)

        # Gaussian fit
        feat['gauss_width_x'], feat['gauss_width_y'], feat['gauss_chi2'] = self.gaussfit()

        # Template fits
        for myclass in CLASSES:
            mu = np.load(os.path.join(PATH_DATA, 'templates', myclass+'_2_mean.npy'))
            sigma = np.load(os.path.join(PATH_DATA, 'templates', myclass+'_2_std.npy'))
            feat['template_'+myclass] = np.mean((mu - self.cutouts_grey[2])**2 / sigma**2)

        return feat

    def summary(self):
        print self.GalaxyID
        print '--------------------'
        print 'Color: {g.red:.0f}, {g.green:.0f}, {g.blue:.0f}'.format(g=self)
        print 'Std: {g.std:.2f}'.format(g=self)
        print 'Gaus: {g.gaussfit}'.format(g=self)


############
# TEMPLATES
############

def save_template(galaxylist, filename_prefix):
    """Computes the average image from a list of galaxyids"""
    log.info('Creating template '+filename_prefix)
    for cutid in [2]:
        images = [Galaxy(galaxyid).cutouts_grey[cutid] for galaxyid in galaxylist]
        np.save(open('{0}_{1}_mean.npy'.format(filename_prefix, cutid), 'wb'),
                np.mean(images, axis=0))
        np.save(open('{0}_{1}_std.npy'.format(filename_prefix, cutid), 'wb'),
                np.std(images, axis=0))

def compute_templates():
    training = Table.read(os.path.join(PATH_DATA, 'training-solutions.fits'))
    for feat in CLASSES:
        args = np.argsort(training[feat])
        save_template(training['GalaxyID'][args[0:1000]],
                      PATH_TEMPLATES+'/'+feat)

def plot_templates():
    for filename in os.listdir(PATH_TEMPLATES):
        if filename.endswith('png'):
            continue
        path = os.path.join(PATH_TEMPLATES, filename)
        target = path+'.png'
        log.info('Creating '+target)

        img = np.load(open(path, 'rb'))
        plt.figure()
        plt.imshow(img)
        plt.savefig(target)
        plt.close()


#####################
# FEATURE GENERATION
#####################

def compute_features_one(galaxyid):
    return Galaxy(galaxyid).features()

def compute_training_features():
    """Writes the solutions and features of the training data to a table."""
    training = Table.read(os.path.join(PATH_DATA, 'training-solutions.fits'))
    pool = multiprocessing.Pool()
    job = pool.imap_unordered(compute_features_one, training['GalaxyID'])
    features = []
    for i, feat in enumerate(job, 1):
        features.append(feat)
        if i % 50 == 0:
            print "Done {0:.1f}%".format(100.*i/float(training['GalaxyID'].size))
    # Write results
    mytable = join(Table(features), training, keys=['GalaxyID'])
    mytable.write(os.path.join(PATH_DATA, 'features-training.fits'),
                  overwrite=True)

def compute_benchmark_features():
    """Writes the features of the benchmark data to a table."""
    pool = multiprocessing.Pool()
    bm = np.load(os.path.join(PATH_DATA, 'benchmark-zeros.npy'))

    job = pool.imap_unordered(compute_features_one, bm['galaxyid'])
    features = []
    for i, feat in enumerate(job, 1):
        features.append(feat)
        if i % 50 == 0:
            print "Done {0:.1f}%".format(100.*i/float(bm['galaxyid'].size))
    # Write results
    Table(features).write(os.path.join(PATH_DATA, 'features-benchmark.fits'),
                          overwrite=True)


if __name__ == '__main__':
    """
    GalaxyIDs = [100859, 124265, 306082, 449965, 564831, 708364, 805284, 951044,
                 105447, 267783, 321044, 442980, 536591, 678853, 959508, 740080]

    plt.figure()
    plt.interactive(True)
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, wspace=0.1, hspace=0.1)

    for i, gid in enumerate(GalaxyIDs):
        cutid = 2
        galaxy = Galaxy(gid)
        #galaxy.summary()
        plt.subplot(4, 4, i+1)
        plt.imshow(galaxy.cutouts_grey[cutid], interpolation='nearest')
        plt.title('{0:.3f}'.format(galaxy.vsym(cutid)), fontsize=20)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.tight_layout()
        #plt.imshow(shrink(np.fliplr(galaxy.cutouts_grey[2][:,50:]), 10, 5), interpolation='nearest')

    plt.show()
    #plt.close()
    """
    #compute_templates()
    #plot_templates()
    compute_training_features()
    #compute_benchmark_features()
    