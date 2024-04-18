import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import expon
import numpy as np

data = pd.read_csv(sys.argv[1], sep = ' ')

plt.xlabel("value of loss")

#sns.histplot(data,  bins = 300)
a  = np.random.default_rng().normal(0, 300, size=1000000)
a = [x/a.max()*300 for x in a]
sns.histplot(data, bins = 300)
sns.histplot(a, bins = 200)
plt.show()

"""sns.kdeplot(data)

a  = np.random.default_rng().standard_t(10, size=500)
sns.kdeplot(a)
ƒ
plt.show()"""