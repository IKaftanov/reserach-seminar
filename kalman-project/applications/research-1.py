import os
import logging
import numpy as np
import matplotlib.pyplot as plt


from kalman import KalmanFilter1D, find_optimal_parameters

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

np.random.seed(100)

y = np.sin(np.linspace(0, 100, 5000))
z = np.random.normal(0, 1, 5000)

z_y = z + y

# filterr = KalmanFilter1D(data=z_y, q=z_y.std(), r=10)

# filterr.fit()
err, params = find_optimal_parameters(incoming_data=z_y, real_data=y, grid_size=1)

params['data'] = z_y

fil = KalmanFilter1D(**params)
fil.fit()


ax = fil.plot(fig_size=(20, 5))
ax.plot(y, label='real data', color='magenta')

plt.legend()
plt.grid()
plt.savefig(f'figures\\{os.path.basename(__file__).split(".")[0]}_test.png', format='png', dpi=150, quality=100)
plt.show()
