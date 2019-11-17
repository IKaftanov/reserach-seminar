import numpy as np
import matplotlib.pyplot as plt

import logging


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
        k_gain = self.H * self.p_est[-1] / (self.H * self.p_est[-1] * self.H + self.R)
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
        return np.mean(np.power(np.sum(real_data - self.states), 2))

    def _pack_variables(self):
        return {'F': self.F, 'B': self.B, 'H': self.H, 'Q': self.Q, 'R': self.R}

    def __str__(self):
        return ' '.join([f'{key} = {value}' for key, value in self._pack_variables().items()])


def find_optimal_parameters(incoming_data: np.ndarray, real_data: np.ndarray, **kwargs):
    grid_size = kwargs.get('grid_size', 10)

    selector_length = kwargs.get('selector_length', incoming_data.shape[0])
    if real_data.shape[0] < selector_length:
        selector_length = real_data.shape[0]
        logging.info(f'You are using only {selector_length} of existing set')

    real_data = real_data[:selector_length].copy()
    incoming_data = incoming_data[:selector_length].copy()

    r_bound = np.mean(np.abs(real_data - incoming_data)) * 10000
    r_grid = np.linspace(-r_bound, r_bound, grid_size)
    q_bound = real_data.std()
    q_grid = np.linspace(-q_bound, q_bound, grid_size)
    f_grid = np.linspace(1, 1, 1)
    b_grid = np.linspace(0, 0, 1)
    h_grid = np.linspace(1, 1, 1)
    _product = np.array(np.meshgrid(r_grid, q_grid, f_grid, b_grid, h_grid)).T.reshape(-1, 5)
    _optimal_set = None
    _min_error = 10e+5
    for i, _set in enumerate(_product):
        logging.info(f'perform {i+1} epoch of {_product.shape[0]}x{_product.shape[1]}')
        _filter = KalmanFilter1D(data=incoming_data, q=_set[1], r=_set[0], f=_set[2], b=_set[3], h=_set[4])
        _filter.fit()
        if _filter.get_delta(real_data) < _min_error:
            _optimal_set = _set.copy()
    return _min_error, {key: value for key, value in zip(['r', 'q', 'f', 'b', 'h'], _optimal_set)}
