import numpy as np
import pandas as pd
import matplotlib as plot
df = pd.read_csv('pants.csv')
hist=df.plot.hist(bins=70) 
plot.pyplot.show()
