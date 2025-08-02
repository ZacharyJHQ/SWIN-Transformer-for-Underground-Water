import numpy as np

class RandomGroundEnv:
    # ground info
    height = 5000.0  # meter
    width = 5000.0  # meter
    num_row = 100
    num_col = 100
    delta_row = height / num_row
    delta_col = width / num_col

    z_top = np.random.random() * 20 - 10  # meter
    z_delta = np.random.random() * 90 + 10  # meter
    z_bottom = z_top - z_delta
    num_layer = 1
    bottom_arr = np.linspace(z_top, z_bottom, num_layer + 1)

    # 边界水头
    boundary = np.ones((num_layer, num_row, num_col), dtype=np.int32)
    boundary[:, :, 0] = -1
    boundary[:, :, -1] = -1
    boundary[:, 0, 1:-1] = -1
    starting_head = np.ones((num_layer, num_row, num_col), dtype=np.float32)
    starting_head[:, :, 0] = np.random.random() * 3 + 1
    starting_head[:, :, -1] = np.random.random() * 3 + 1
    starting_head[:, 0, 1:-1] = np.random.random() * 3 + 1

    # 含水层
    seed_num = np.random.randint(2, 6)
    seed_pos = np.random.randint(100, size=(seed_num, 2))
    seed_coeff = np.random.random(size=(seed_num)) * 9 + 1
    layer_property = np.ones((num_layer, num_row, num_col), dtype=np.float32)
    for r in range(num_row):
        for c in range(num_col):
            _pos = np.array([r, c])
            _distances = np.sum(np.abs(_pos - seed_pos), axis=1)
            _nearest = np.argmin(_distances)
            layer_property[:, r, c] = seed_coeff[_nearest]

    # well
    well_num = np.random.randint(1, 6)
    well_stress_data = {
        0: [
            [
                0, 
                np.random.randint(100), 
                np.random.randint(100), 
                np.random.random() * 1200 - 600
            ] for _ in range(well_num)
        ]
    }

    # head
    head = None

