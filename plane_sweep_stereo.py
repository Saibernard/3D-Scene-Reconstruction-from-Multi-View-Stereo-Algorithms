import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane using intrinsics and extrinsics
    
    Hint:
    depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points, here 2x2 correspinds to 4 corners
    """

    points = np.array(
        (
            (0, 0, 1),
            (width, 0, 1),
            (0, height, 1),
            (width, height, 1),
        ),
        dtype=np.float32,
    ).reshape(2, 2, 3)

    """ YOUR CODE HERE
    """
    print("K", K)
    print("width", width)
    print("height", height)
    print("depth", depth)
    print("Rt", Rt)

    points = points.reshape(4,3)
    uncalibrated_points = np.linalg.inv(K) @ points.T
    #normalising
    uncalibrated_points = uncalibrated_points/uncalibrated_points[-1,:]
    R = Rt[:,0:3]
    T = Rt[:,3].reshape(-1,1)

    points = ((R.T @((depth*uncalibrated_points)-T)).T).reshape(2,2,3)

    """ END YOUR CODE
    """
    return points


def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    
    Hint:
    Z * projections = K @ T @ p, where p is the input points and projections is the output, T is the 4x4 matrix of Rt (3x4)
    
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """

    H = points.shape[0]
    W = points.shape[1]
    print("H", H)
    print("W", W)
    points = points.reshape((H * W, 3))
    ones = np.ones((H * W, 1))
    points = np.hstack((points, ones))
    projected_points = (K @ Rt @ points.T)
    print("projected points", projected_points)
    projected_points = projected_points.T/projected_points[-1,:].reshape(-1,1)
    points = projected_points[:,:-1].reshape((H,W,2))


    """ END YOUR CODE
    """
    return points


def warp_neighbor_to_ref(
    backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor
):
    """
    Warp the neighbor view into the reference view
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective
    
    ! Note, when you use cv2.warpPerspective, you should use the shape (width, height), NOT (height, width)
    
    Hint: you should do the follows:
    1.) apply backproject_corners on ref view to get the virtual 3D corner points in the virtual plane
    2.) apply project_fn to project these virtual 3D corner points back to ref and neighbor views
    3.) use findHomography to get teh H between neighbor and ref
    4.) warp the neighbor view into the reference view

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width= neighbor_rgb.shape[:2]
    print(height, width)
    # print(neighbor_rgb)

    D = neighbor_rgb.shape[2]
    src_pts = np.array(((0,0),(width,0),(0,height),(width,height)),dtype=np.float32)

    i_view = backproject_fn(K_ref, width, height, depth, Rt_ref)
    j_view = project_fn(K_neighbor,Rt_neighbor,i_view)
    j_Height = j_view.shape[0]
    j_Width = j_view.shape[1]

    print(j_Height)
    print(j_Width)

    distance_pts = j_view.reshape((j_Height* j_Width, 2))
    M, Mask = cv2.findHomography(src_pts, distance_pts, cv2.RANSAC)
    M_inv = np.linalg.inv(M)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, M_inv, dsize = (width, height))




    """ YOUR CODE HERE
    """

    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """
    Compute the cost map between src and dst patchified images via the ZNCC metric

    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value,
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """

    M = src.shape[0]
    N = src.shape[1]
    W_1_mean = np.mean(src, axis=2)
    W_2_mean = np.mean(dst, axis=2)

    zncc = np.zeros((M,N,3))

    sigma_w1 = np.std(src, axis=2)
    sigma_w2 = np.std(dst, axis = 2)

    for i in range(M):
        for j in range(N):
            zncc[i,j,0] = np.sum((src[i,j,:,0] - W_1_mean[i,j,0]) * (dst[i,j,:,0] -
                            W_2_mean[i,j,0])) / ((sigma_w1[i,j,0] * sigma_w2[i,j,0]) + EPS)
            zncc[i, j, 1] = np.sum((src[i, j, :, 1] - W_1_mean[i, j, 1]) * (dst[i, j, :, 1] -
                                                                            W_2_mean[i, j, 1])) / (
                                        (sigma_w1[i, j, 1] * sigma_w2[i, j, 1]) + EPS)
            zncc[i, j, 2] = np.sum((src[i, j, :, 2] - W_1_mean[i, j, 2]) * (dst[i, j, :, 2] -
                                                                            W_2_mean[i, j, 2])) / (
                                        (sigma_w1[i, j, 2] * sigma_w2[i, j, 2]) + EPS)

    zncc = np.sum(zncc, axis = 2).reshape((zncc.shape[0], zncc.shape[1]))

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """

    f = K[1,1]
    f11 = K[0,0]
    f22 = K[1,1]
    u0 = K[0,-1]
    v0 = K[1,-1]

    r_1 = ((_u - u0) * dep_map)/f11
    g_2 = ((_v - v0) * dep_map)/ f22

    print(r_1, g_2)
    print(np.shape(r_1))
    print(np.shape(g_2))

    xyz_cam = np.dstack((r_1, g_2, dep_map))

    """ END YOUR CODE
    """
    return xyz_cam
