import itertools

import numpy as np

from scipy.ndimage import spline_filter
from scipy.ndimage.morphology import distance_transform_edt

def perform_voting(patches, output_shape, expected_shape, extraction_step, window_sep=(8, 8, 8)) :
    vote_img = np.zeros(expected_shape)
    vote_count = np.zeros(expected_shape)

    coordinates = generate_indexes(
        output_shape, extraction_step, expected_shape)

    W = np.ones(output_shape)
    W[window_sep[0]:-window_sep[0], window_sep[1]:-window_sep[1], window_sep[2]:-window_sep[2]] = 0

    W_dist = distance_transform_edt(W)
    W_dist = 1-W_dist/W_dist.max()

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        vote_img[selection] += np.multiply(patches[count], W_dist)
        vote_count[selection] += np.multiply(np.ones(vote_img[selection].shape), W_dist)
        
    vote_count[vote_count == 0] = 1

    return spline_filter(np.divide(vote_img, vote_count))

def perform_voting_probabilities(patches, output_shape, expected_shape, extraction_step, num_classes) :
    vote_img = np.zeros(expected_shape + (num_classes, ))

    coordinates = generate_indexes(
        output_shape, extraction_step, expected_shape)

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        selection += [slice(None)]
        vote_img[selection] += patches[count]

    return np.argmax(vote_img, axis=3)

def generate_indexes(output_shape, extraction_step, expected_shape) :
    ndims = len(output_shape)

    poss_shape = [output_shape[i] + extraction_step[i] * ((expected_shape[i] - output_shape[i]) // extraction_step[i]) for i in range(ndims)]

    idxs = [range(output_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]
    
    return itertools.product(*idxs)