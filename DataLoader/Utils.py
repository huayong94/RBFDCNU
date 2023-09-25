import numpy as np
import math
# np.random.seed(1234)


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def crop(img, center, vol_size):
    center_x, center_y = center
    return img[center_y - int(vol_size[0] / 2):center_y + int(vol_size[0] / 2),
               center_x - int(vol_size[1] / 2):center_x + int(vol_size[1] / 2)]


def randomTransform(degree_interval, translate_interval, scale_interval,
                    image_size):
    degree = np.random.uniform(degree_interval[0], degree_interval[1])
    degree = degree * math.pi / 180
    scale = np.random.uniform(scale_interval[0], scale_interval[1])
    tx = np.random.uniform(translate_interval[0],
                           translate_interval[1]) / image_size[1] * 2
    ty = np.random.uniform(translate_interval[0],
                           translate_interval[1]) / image_size[0] * 2

    rm = np.array([[math.cos(degree), -math.sin(degree), 0],
                   [math.sin(degree), math.cos(degree), 0], [0, 0, 1]])
    tm = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    sm = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]])
    # combine the transforms
    m = np.matmul(sm, np.matmul(tm, rm))
    # remove the last row; it's not used by affine transform
    return m[0:2, 0:3]
