import numpy as np
import logging
from .filter import KalmanFilter1D


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