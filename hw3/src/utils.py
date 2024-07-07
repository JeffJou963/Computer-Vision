import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A: N=4, A=[8x9]
    A = np.zeros((2*N,9))
   
    for i in range (N):
        A[2*i,:] = [u[i,0], u[i,1], 1, 0, 0, 0, -u[i,0]*v[i,0], -u[i,1]*v[i,0], -v[i,0]]
        A[2*i+1,:] = [0, 0, 0, u[i,0], u[i,1], 1, -u[i,0]*v[i,1], -u[i,1]*v[i,1], -v[i,1]]

    # TODO: 2.solve H with A
    # [8x8][8][9x9]
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    yy, xx = np.meshgrid(range(ymin, ymax), range(xmin,xmax), indexing='ij')

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate --> NxNx3
    coords = np.stack([xx, yy, np.ones_like(xx)], axis=-1)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        coords_src = coords @ H_inv.T
        coords_src /= coords_src[..., 2:3]
        x_src, y_src = coords_src[..., 0], coords_src[..., 1]

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        x_src, y_src = np.round(x_src).astype(int), np.round(y_src).astype(int)
        mask = (x_src >= 0) & (x_src < w_src) & (y_src >= 0) & (y_src < h_src)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        x_src, y_src = x_src[mask], y_src[mask]

        # TODO: 6. assign to destination image with proper masking
        dst[yy[mask], xx[mask]] = src[y_src, x_src]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        coords_dst = coords @ H.T
        coords_dst /= coords_dst[..., 2:3]
        x_dst ,y_dst = coords_dst[..., 0], coords_dst[..., 1]        

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        x_dst, y_dst = np.round(x_dst).astype(int), np.round(y_dst).astype(int)
        mask = (x_dst >= 0) & (x_dst < w_dst) & (y_dst >= 0) & (y_dst < h_dst)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        x_dst, y_dst = x_dst[mask], y_dst[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[y_dst, x_dst] = src[yy[mask], xx[mask]]


    return dst 

