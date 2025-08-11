
from typing import Any, Dict

import numpy as np


class RandomGroundEnv:
    def __init__(self):
        # ground info
        self.height = 6400.0  # meter
        self.width = 6400.0  # meter
        self.num_row = 128
        self.num_col = 128
        self.delta_row = self.height / self.num_row
        self.delta_col = self.width / self.num_col

        self.z_top = np.random.random() * 20 - 10  # meter
        self.z_delta = np.random.random() * 90 + 10  # meter
        self.z_bottom = self.z_top - self.z_delta
        self.num_layer = 1
        self.bottom_arr = np.linspace(self.z_top, self.z_bottom, self.num_layer + 1)

        # 边界水头
        self.boundary = np.ones((self.num_layer, self.num_row, self.num_col), dtype=np.int32)
        self.boundary[:, :, 0] = -1
        self.boundary[:, :, -1] = -1
        self.boundary[:, 0, 1:-1] = -1
        self.starting_head = np.ones((self.num_layer, self.num_row, self.num_col), dtype=np.float32)
        self.starting_head[:, :, 0] = np.random.random() * 3 + 1
        self.starting_head[:, :, -1] = np.random.random() * 3 + 1
        self.starting_head[:, 0, 1:-1] = np.random.random() * 3 + 1

        # 含水层
        self.seed_num = np.random.randint(2, 6)
        self.seed_pos = np.random.randint(100, size=(self.seed_num, 2))
        self.seed_coeff = np.random.random(size=(self.seed_num)) * 9 + 1
        self.layer_property = np.ones((self.num_layer, self.num_row, self.num_col), dtype=np.float32)
        for r in range(self.num_row):
            for c in range(self.num_col):
                _pos = np.array([r, c])
                _distances = np.sum(np.abs(_pos - self.seed_pos), axis=1)
                _nearest = np.argmin(_distances)
                self.layer_property[:, r, c] = self.seed_coeff[_nearest]

        # well
        self.well_num = np.random.randint(1, 6)
        self.well_stress_data = {
            0: [
                [
                    0, 
                    np.random.randint(100), 
                    np.random.randint(100), 
                    np.random.random() * 1200 - 600
                ] for _ in range(self.well_num)
            ]
        }

        # head
        self.head = None

    def to_dict(self) -> Dict[str, Any]:
        attributes = [
            "height",
            "width",
            "num_row",
            "num_col",
            "delta_row",
            "delta_col",
            "z_top",
            "z_delta",
            "z_bottom",
            "num_layer",
            "bottom_arr",
            "boundary",
            "starting_head",
            "seed_num",
            "seed_pos",
            "seed_coeff",
            "layer_property",
            "well_num",
            "well_stress_data",
            "head",
        ]
        d = {}
        for _attr in attributes:
            d[_attr] = self.__getattribute__(_attr)
        return d

