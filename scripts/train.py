import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcess
from astropy.table import Table

training_data = Table.read('features/features-training.fits')


feature_names = ['gauss_chi2', 'gauss_width_x', 'gauss_width_y']
for name in ['min', 'max', 'std', 'std2', 'p10', 'p50', 'p90',
             'red', 'green', 'blue', 'hsym', 'vsym']: 
    for size in [10, 40, 100]:
        feature_names.append('{0}_{1}'.format(name, size))



training_features = training_data[feature_names]
training_features['col1'] = training_features['red_10'] - training_features['green_10']
training_features['col2'] = training_features['green_10'] - training_features['blue_10']
training_features['col3'] = training_features['red_10'] - training_features['blue_10']
training_features['col4'] = training_features['red_40'] - training_features['green_40']
training_features['col5'] = training_features['green_40'] - training_features['blue_40']
training_features['col6'] = training_features['red_40'] - training_features['blue_40']
training_features['col7'] = training_features['red_100'] - training_features['green_100']
training_features['col8'] = training_features['green_100'] - training_features['blue_100']
training_features['col9'] = training_features['red_100'] - training_features['blue_100']

target_names = ['Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2',
                 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1',
                 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2',
                 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2',
                 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7',
                 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2',
                 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4',
                 'Class11.5', 'Class11.6']
targets = training_data[target_names]

X = np.array(training_features.__array__().tolist())
Y = np.array(targets.__array__().tolist())

algorithm = RandomForestRegressor(n_jobs=-1, n_estimators=200, verbose=100)

idx = np.arange(len(X))
np.random.shuffle(idx)
n_validate = 4000  # How many samples to keep for validation?
idx_training = idx[n_validate:]
algorithm.fit(X[idx_training], Y[idx_training])

idx_validate = idx[:n_validate]
pred = algorithm.predict(X[idx_validate])
score = np.sqrt(np.mean(np.power(pred - Y[idx_validate], 2)))


# Importances
z = zip(feature_names, algorithm.feature_importances_)
z.sort(key = lambda t: t[1], reverse=True)
for a, b in z:
    print '{0} {1:.3f}'.format(a, b)

print 'RMS = {0}'.format(score)



"""
# Produce benchmark result
benchmark_data = Table.read('features/features-benchmark.fits')
benchmark_features = benchmark_data[feature_names]

benchmark_features['col1'] = benchmark_features['core_red'] / benchmark_features['core_blue']
benchmark_features['col2'] = benchmark_features['core_green'] / benchmark_features['core_blue']
benchmark_features['col3'] = benchmark_features['core_red'] / benchmark_features['core_green']
benchmark_features['col4'] = benchmark_features['blue'] / benchmark_features['red']
benchmark_features['col5'] = benchmark_features['blue'] / benchmark_features['green']
benchmark_features['col6'] = benchmark_features['green'] / benchmark_features['red']


X2 = np.array(benchmark_features.__array__().tolist())

t = Table(algorithm.predict(X2), names=target_names)
t.add_column(benchmark_data['GalaxyID'], index=0)
t.write('benchmark.csv')
"""