import numpy
from chainer import cuda

def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=numpy.int32,
                            distances_dtype=numpy.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices, distances

import numpy as np
import sys

def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines

def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True