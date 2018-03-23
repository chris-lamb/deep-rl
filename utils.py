import numpy as np
import cv2


def preprocess_state(state):
    # preprocess 210x160x3 uint8 frame into 80x80 2D float array
    state = np.array(state[35:195])  # crop to 160 x 160 x 3
    state = convert_rgb_to_grayscale(state)
    state = cv2.resize(state, (80, 80))  # downsample by factor of 2
    return state


def convert_rgb_to_grayscale(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray.astype(float).reshape(160, 160)
