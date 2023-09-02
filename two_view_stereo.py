import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data

# from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    rgb_i_rect = cv2.warpPerspective(rgb_i, M = K_i_corr@R_irect@np.linalg.inv(K_i), dsize = (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, M = K_j_corr@R_jrect@np.linalg.inv(K_j), dsize = (w_max, h_max))

    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    """Student Code Starts"""
    p_i = np.zeros((3,1))
    p_j = np.zeros((3,1))

    p_w = p_j - T_wj
    # print(p_j)
    # print(p_w)
    p_i_w = np.linalg.inv(R_wi)@(p_i-T_wi)
    p_j_w = np.linalg.inv(R_wj)@(p_j - T_wj)

    B = np.linalg.norm(p_i_w - p_j_w)

    R_ji = R_wi@R_wj.T

    p_i = R_wi@p_w + T_wi
    p_j = R_wj@p_w+ T_wj
    T_ji = p_i - (R_ji@p_j)




    """Student Code Ends"""

    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ji : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)
    
    # ! Note, we define a small EPS at the beginning of this file, use it when you normalize each column

    """Student Code Starts"""

    r2 = e_i/ np.linalg.norm(e_i)
    print(r2)

    # y_epi = np.array([[0],[1],[0]])
    z_axis = np.array([[0],[0],[1]])
    # print(np.shape(z_axis))
    r1 = np.cross(r2, z_axis.reshape(-1))
    r1 = r1/np.linalg.norm(r1)
    r3 = np.cross(r1,r2)
    r3 = r3/np.linalg.norm(r3)
    R_irect = np.vstack((r1,r2,r3))
    """Student Code Ends"""

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]
    # print(M, N)

    ssd = np.zeros((M, N, 3))

    for i in range(M):
        for j in range(N):
            ssd[i, j, 0] = np.square(src[i, :, 0] - dst[j, :, 0]).sum()
            ssd[i, j, 1] = np.square(src[i, :, 1] - dst[j, :, 1]).sum()
            ssd[i, j, 2] = np.square(src[i, :, 2] - dst[j, :, 2]).sum()
            # print(ssd)

    """Student Code Ends"""
    ssd = np.sum(ssd, axis=2)

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SAD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]

    sad = np.zeros((M, N, 3))

    for i in range(M):
        for j in range(N):
            sad[i, j, 0] = np.abs(src[i, :, 0] - dst[j, :, 0]).sum()
            sad[i, j, 1] = np.abs(src[i, :, 1] - dst[j, :, 1]).sum()
            sad[i, j, 2] = np.abs(src[i, :, 2] - dst[j, :, 2]).sum()

    sad = np.sum(sad, axis=2)

    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    M = src.shape[0]
    N = dst.shape[0]

    W_1_mean = np.mean(src, axis=1)
    W_2_mean = np.mean(dst, axis=1)
    sigma_w1 = np.zeros((M, 1, 3))
    sigma_w2 = np.zeros((N, 1, 3))
    zncc = np.zeros((M,N,3))

    for i in range(M):
        sigma_w1[i, :, 0] = np.sqrt(np.mean(np.square(src[i, :, 0] - W_1_mean[i,0])))
        sigma_w1[i, :, 1] = np.sqrt(np.mean(np.square(src[i, :, 1] - W_1_mean[i, 1])))
        sigma_w1[i, :, 2] = np.sqrt(np.mean(np.square(src[i, :, 2] - W_1_mean[i, 2])))

    for j in range(N):
        sigma_w2[j, :, 0] = np.sqrt(np.mean(np.square(dst[j, :, 0] - W_2_mean[j, 0])))
        sigma_w2[j, :, 1] = np.sqrt(np.mean(np.square(dst[j, :, 1] - W_2_mean[j, 1])))
        sigma_w2[j, :, 2] = np.sqrt(np.mean(np.square(dst[j, :, 2] - W_2_mean[j, 2])))

    for i in range(M):
        for j in range(N):
            zncc[i,j,0] = np.sum((src[i,:,0] - W_1_mean[i,0])* (dst[j,:,0]
                            - W_2_mean[j,0]))/((sigma_w1[i,:,0]* sigma_w2[j,:,0]) + EPS)
            zncc[i, j, 1] = np.sum((src[i, :, 1] - W_1_mean[i, 1]) * (dst[j, :, 1]
                                - W_2_mean[j, 1])) / ((sigma_w1[i, :, 1] * sigma_w2[j, :, 1]) + EPS)
            zncc[i, j, 2] = np.sum((src[i, :, 2] - W_1_mean[i, 2]) * (dst[j, :, 2]
                            - W_2_mean[j, 2])) / ((sigma_w1[i, :, 2] * sigma_w2[j, :, 2]) + EPS)

    zncc = np.sum(zncc, axis = 2)


    """Student Code Ends"""

    # ! note here we use minus zncc since we use argmin outside, but the zncc is a similarity, which should be maximized
    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    H, W = image.shape[0:2]

    padded_image = np.pad(image[:,:,0], k_size//2)
    padded_image = np.dstack((padded_image, np.pad(image[:,:,1], k_size//2)))
    padded_image = np.dstack((padded_image, np.pad(image[:, :, 2], k_size // 2)))

    pad_size = k_size//2
    patch_buffer = np.zeros((H,W, k_size**2,3))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch_buffer[i, j, :, 0] = padded_image[i:i + k_size, j:j + k_size,0].flatten()
            patch_buffer[i, j, :, 1] = padded_image[i:i + k_size, j:j + k_size, 1].flatten()
            patch_buffer[i, j, :, 2] = padded_image[i:i + k_size, j:j + k_size, 2].flatten()
    """Student Code Starts"""

    # print(patch_buffer)

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(
    rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch
):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func : function, optional
        this is for auto-grader purpose, in grading, we will use our correct implementation of the image2path function to exclude double count for errors in image2patch function

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    H = rgb_i.shape[0]
    W = rgb_j.shape[1]
    disp_map = np.zeros((H,W), dtype=np.float64)
    lr_consistency_mask = np.zeros((H,W), dtype=np.float64)

    patches_i = image2patch(rgb_i.astype(float) / 255.0, k_size)
    patches_j = image2patch(rgb_j.astype(float)/ 255.0, k_size)

    vi_idx = np.arange(H)
    vj_idx = vi_idx

    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    valid_disp_mask = disp_candidates > 0.0

    for i in range(rgb_j.shape[1]):
        buf_i, buf_j = patches_i[:,i].astype(np.float64), patches_j[:,i].astype(np.float64)
        value = kernel_func(buf_i, buf_j)
        upper = value.max() + 1.0
        value[~valid_disp_mask] = upper

        best_matched_right_pixel = np.argmin(value, axis=1)
        best_matched_left_pixel = np.argmin(value[:, best_matched_right_pixel], axis=0)
        # print(v)
        # print(best_matched_left_pixel)
        consistent_flag = best_matched_left_pixel == vi_idx
        # print(consistent_flag)
        disp_map[:, i] = disp_candidates[np.arange(H), best_matched_right_pixel]
        lr_consistency_mask[:, i] = consistent_flag
    """Student Code Ends"""

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    x = np.arange(disp_map.shape[1])
    y = np.arange(disp_map.shape[0])

    xx,yy = np.meshgrid(x,y)
    f = K[1,1]
    dep_map = f*(B/disp_map)
    u0 = K[0, -1]
    v0 = K[1,-1]
    f1 = K[0,0]
    f2 = K[1,1]

    xyz_cam = np.dstack(((xx - u0) * dep_map/f1, (yy - v0) * dep_map/f2, dep_map))


    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is:
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""

    R_cw = R_wc.T
    T_cw = -R_wc.T @ T_wc
    pcl_world = (R_cw@pcl_cam.T + T_cw).T

    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
