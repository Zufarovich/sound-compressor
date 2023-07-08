import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np

stat = open("test.txt")

data =[abs(float(x)) for x in stat.readlines()]
lst = []
for el in data:
    if el != 0.0:
        lst.append(el)
plt.grid(color='green', linestyle='dotted', linewidth=0.5)
plt.minorticks_on()
matplotlib.scale.LogScale(matplotlib.axis.YAxis, base = 2)

#plt.hist(np.random.exponential(np.mean(lst), len(lst) - 10000), bins = int(1.73*len(lst)**(1/3)))
#plt.hist(np.random.geometric(1/np.mean(lst), len(lst) - 100000), bins = int(1.73*len(lst)**(1/3)))
plt.hist(lst, bins = int(1.73*len(lst)**(1/3)))
plt.show()
