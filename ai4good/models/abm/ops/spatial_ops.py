import numpy as np
from numba import njit

# this defines the size of the camp in coordinate plane.
CAMP_X = 1.0
CAMP_Y = 1.0


@njit
def assign_block(pos, grid_size):
    """
    Assign households to equidistant blocks based on a hypothetical grid size.
    Parameters
    ----------
        pos : array_like
            (hb+ht, 2) array containing (x, y) position of iso-boxes and tents
        grid_size : tuple or list or array_like
            (n, m) tuple containing grid size

    Returns
    -------
        location : array_like
            A 2D array containing (x, y) coordinates blocks formed by the grid
        label: array_like
            A 1D array containing block numbers for each household
        shared : array_like
            A 2D boolean array of shared blocks at the household level.
            shared[i, j] will be `True` if household i and j share the toilet or food line else `False`
    """

    # for the hypothetical grid in the camp of size (CAMP_X, CAMP_Y), calculate the initial (bottom left) coordinates
    # of the toilet/foodline based on grid size
    x0 = CAMP_X / (2 * grid_size[0])
    y0 = CAMP_Y / (2 * grid_size[1])

    # total number of blocks
    total_blocks = grid_size[0] * grid_size[1]

    # 2D location array containing location of the blocks formed by the grid.
    location = np.zeros(shape=(total_blocks, 2), dtype=np.float32)

    # 1D label array for each household
    label = np.zeros(shape=(pos.shape[0],), dtype=np.int32)

    # 2D matrix containing 1/0 if block (i, j) is shared by both i and j
    shared = np.zeros(shape=(pos.shape[0], pos.shape[0]), dtype=np.int32)

    # iteratively calculate the position of each toilet/foodline.
    # Note: these places are placed uniformly in the grid.
    counter = 0
    for j in range(grid_size[1]):
        # position of the y coordinate
        posy = y0 + j * (CAMP_Y / grid_size[1])
        for i in range(grid_size[0]):
            # position of the x coordinate
            posx = x0 + i * (CAMP_X / grid_size[0])
            # store block center location
            location[counter, 0] = posx
            location[counter, 1] = posy
            # increase the counter to move to next block
            counter += 1

    # calculate block labels for the households. The household at pos (x, y) belonging to block b will have label b
    for i in range(pos.shape[0]):
        # calculate the block label based on household coordinates
        label_x = int(pos[i, 0] / (CAMP_X / grid_size[0]))
        label_y = int(pos[i, 1] / (CAMP_Y / grid_size[1]))
        # store block label in `label` array for each household
        label[i] = label_y * grid_size[0] + label_x + 1

    # if two households share same block label, store 1 in `shared` array else 0
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            shared[i, j] = 1 if label[i] == label[j] else 0

    # return results
    return location, label, shared


@njit
def distance_matrix(d):
    """
    Calculates distance matrix from 2D coordinates.

    Parameters
    ----------
        d : array_like
            (n, 2) array containing (x, y) coordinates for n points.

    Returns
    -------
        out : array_like
            (n, n) distance matrix where element (i, j) stores distance between point (i) and point (j).

    """

    n = d.shape[0]  # number of points
    mat = np.zeros(shape=(n, n), dtype=np.float32)  # initialize distance matrix

    # loop through all pairs
    for i in range(n):
        for j in range(n):
            # calculate Euclidean distance between (i) and (j)
            dij = (d[i, 0] - d[j, 0]) ** 2 + (d[i, 1] - d[j, 1]) ** 2
            dij = dij ** 0.5
            # store result in distance matrix
            mat[i, j] = dij

    return mat
