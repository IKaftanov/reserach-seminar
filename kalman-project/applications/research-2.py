import os
import logging
import numpy as np
import matplotlib.pyplot as plt


from kalman import KalmanFilter1D, find_optimal_parameters

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

np.random.seed(100)

y = np.linspace(-1, 1, 5000)
z = np.random.normal(0, 1, 5000)

z_y = z + y


fil = KalmanFilter1D(data=z_y, q=z_y.std(), r=1000)
fil.fit()


ax = fil.plot(fig_size=(20, 5))
ax.plot(y, label='real data', color='magenta')
plt.legend()
plt.grid()
plt.savefig(f'figures\\{os.path.basename(__file__).split(".")[0]}_test.png', format='png', dpi=150, quality=100)
plt.show()
