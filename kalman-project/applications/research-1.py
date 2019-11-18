import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from kalman import KalmanFilter1D, find_optimal_parameters

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


np.random.seed(100)

y = np.linspace(0, 1, 5000)
z = np.random.normal(0, 1, 5000)

z_y = z + y

err, params = find_optimal_parameters(incoming_data=z_y, real_data=y, grid_size=10)
params['data'] = z_y

fil = KalmanFilter1D(**params)
fil.fit()
print(fil.get_delta(y))
ax = fil.plot(fig_size=(20, 5))
ax.plot(y, label='real data', color='magenta')
plt.legend()
plt.grid()
plt.savefig(f'figures\\{os.path.basename(__file__).split(".")[0]}_test.png', format='png', dpi=150, quality=100)
plt.show()
