import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter1D:
    __name__ = 'KalmanFilter1D'

    F = 1
    B = 0
    H = 1

    covariances = []
    states = []

    x_est = []
    p_est = []

    is_fitted = False

    def __init__(self, data: np.ndarray, q: float, r: float,
                 f: float = None, b: float = None, h: float = None, **kwargs) -> None:
        if f:
            self.F = f
        if b:
            self.B = b
        if h:
            self.H = h

        self.Q = q
        self.R = r

        self.data = data.copy()

        self._set_state(*kwargs.get('init_states', (data[0], 0.1)))

    def _set_state(self, state, covariance):
        self.states.append(state)
        self.covariances.append(covariance)

    def _time_update(self):
        if not self.states or not self.covariances:
            raise Exception('<self.states> or <self.covariances> are not defined')
        self.x_est.append(self.F * self.states[-1])
        self.p_est.append(self.F * self.covariances[-1] * self.F + self.Q)

    def _measurement_update(self, z):
        if not self.p_est or not self.x_est:
            raise Exception('<self.p_est> or <self.x_est> are not defined')
        k_gain = self.H * self.p_est[-1] / (self.H * self.p_est[-1] * self.H + self.R)
        _state = self.x_est[-1] + k_gain * (z - self.H * self.x_est[-1])
        _covariance = (1 - k_gain * self.H) * self.p_est[-1]
        self._set_state(_state, _covariance)

    def fit(self):
        for z in self.data:
            self._time_update()
            self._measurement_update(z)

        self.states = np.array(self.states)
        self.covariances = np.array(self.covariances)
        self.x_est = np.array(self.x_est)
        self.p_est = np.array(self.p_est)

        self.is_fitted = True

    def plot(self):
        if self.is_fitted:
            plt.plot(self.data, color='blue', linestyle='solid', label='real data')
            plt.plot(self.states, color='red', linestyle='--', label='states')
            plt.xlabel("t")
            plt.ylabel("z")
            plt.legend()
            plt.grid()
        else:
            raise Exception('You should fit your model first')

    def _pack_variables(self):
        return {'F': self.F, 'B': self.B, 'H': self.H, 'Q': self.Q, 'R': self.R,
                'mean_<x_est>': np.mean(self.x_est),
                'mean_<p_est>': np.mean(self.p_est),
                'mean<states>': np.mean(self.states),
                'mean<covariances>': np.mean(self.covariances)}

    def __str__(self):
        return ''.join([f'{key} = {value}' for key, value in self._pack_variables()])
