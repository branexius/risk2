import numpy as np
import bottleneck as bn
wdw = 1600
price2= np.sort(np.random.randint(1, high=1000, size=100))
btp = bn.move_mean(price2, window=wdw, min_count=2)
