from scipy.stats import binom
import matplotlib.pyplot as plt
import math

n = 50
p = math.pi / 4
results = []
points = range(1, 51)

x = binom(n, p)

for i in points:
    results.append(x.cdf(i))

# make matplotlib figures appear inline in the notebook
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

plt.title('CDF of X from 1 to 50')
plt.xscale('linear')
plt.xlabel('i')
plt.ylabel('P(x<=i)')
plt.plot(points, results)

plt.show()
