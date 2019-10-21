import matplotlib.pyplot as plt
import pathlib as ph
import numpy as np

BASE_PATH = ph.Path(__file__)

DATA_PATH = BASE_PATH.parent.parent.joinpath('data')
all_png = list(DATA_PATH.glob('**/*.png'))
print(all_png)

depth = all_png[0]
file = plt.imread(str(depth))
print(str(depth))
print(file.shape)

plt.show()
print(file.max())
print(file.min())