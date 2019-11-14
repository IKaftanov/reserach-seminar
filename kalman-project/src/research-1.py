import numpy as np
import matplotlib.pyplot as plt


from kalman import KalmanFilter1D

z = np.random.normal(1, 4, 1000)


filterr = KalmanFilter1D(data=z, q=2, r=15)

filterr.fit()
filterr.plot()
plt.show()
