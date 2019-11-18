import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:
    __name__ = 'KalmanFilter1D'

    def __init__(self, data: np.ndarray, q: float, r: float,
                 f: float = None, b: float = None, h: float = None, **kwargs) -> None:
        self.F = f or 1
        self.B = b or 0
        self.H = h or 1
        self.Q = q
        self.R = r
        self.data = data.copy()
        self.init_states = kwargs.get('init_states', (self.data[0], np.cov(self.data)))

        self.covariances = []
        self.states = []
        self.x_est = []
        self.p_est = []
        self.is_fitted = False

    def _set_state(self, state, covariance):
        self.states.append(state)
        self.covariances.append(covariance)

    def _time_update(self):
        if not all(self.states) or not all(self.covariances):
            raise Exception('<self.states> or <self.covariances> are not defined')
        self.x_est.append(self.F * self.states[-1])
        self.p_est.append(self.F * self.covariances[-1] * self.F + self.Q)

    def _measurement_update(self, z):
        if not all(self.p_est) or not all(self.x_est):
            raise Exception('<self.p_est> or <self.x_est> are not defined')
        k_gain = self.H * self.p_est[-1] / (2 * self.H * self.p_est[-1] + self.R)
        _state = self.x_est[-1] + k_gain * (z - self.H * self.x_est[-1])
        _covariance = (1 - k_gain * self.H) * self.p_est[-1]
        self._set_state(_state, _covariance)

    def fit(self):
        self._set_state(*self.init_states)
        for z in self.data:
            self._time_update()
            self._measurement_update(z)

        self.states = np.array(self.states[1:])
        self.covariances = np.array(self.covariances[1:])
        self.x_est = np.array(self.x_est)
        self.p_est = np.array(self.p_est)

        self.is_fitted = True

    def plot(self, **kwargs):
        if self.is_fitted:
            fig_size = kwargs.get('fig_size', (10, 5))
            f, ax = plt.subplots(figsize=fig_size)
            ax.plot(self.data, color='blue', linestyle='solid', label='incoming data')
            ax.plot(self.states, color='red', linestyle='--', label='filtered data')
            ax.set_xlabel('t')
            ax.set_ylabel('z')
            ax.set_title(f'{self.__str__()}')
            return ax
        else:
            raise Exception('You should fit your model first')

    def get_delta(self, real_data: np.ndarray):
        return np.mean(np.sum(real_data - self.states))

    def _pack_variables(self):
        return {'F': self.F, 'B': self.B, 'H': self.H, 'Q': self.Q, 'R': self.R}

    def __str__(self):
        return ' '.join([f'{key} = {value}' for key, value in self._pack_variables().items()])


class KalmanFilterND:
    __name__ = 'KalmanFilterND'
