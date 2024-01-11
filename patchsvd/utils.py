import numpy as np
import cv2
import os
import math
import string
import random

#todo: what to choose for jpeg quality
#todo: per-channel compression vs. image compression

def to_jpeg(image, compression, output_dir, name='temp'):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, f"{name}_jpeg.jpg"), image, (int(cv2.IMWRITE_JPEG_QUALITY), int(compression * 100)))
    img = cv2.imread(os.path.join(output_dir, f"{name}_jpeg.jpg"), cv2.IMREAD_UNCHANGED)
    os.remove(os.path.join(output_dir, f"{name}_jpeg.jpg"))
    return img


def rand_name(size):
    generated_name = ''.join([random.choice(
        string.ascii_letters + string.digits)
        for n in range(size)])
    return generated_name


def extract_rgb(img):
    return np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])


def assemble_rgb(img_r, img_g, img_b):
    shape = (img_r.shape[0], img_r.shape[1], 1)
    return np.concatenate((np.reshape(img_r, shape), np.reshape(img_g, shape), np.reshape(img_b, shape)), axis=2)


def get_extended_window(norm):
    return [[norm, norm, norm], [norm, norm, norm], [norm, norm, norm]]


def multiply_first_k(k, u, smat, vt):
    u = u[:, :min(k, u.shape[1])]
    smat = smat[:min(k, u.shape[1]), :min(k, u.shape[1])]
    vt = vt[:min(k, u.shape[1]), :]
    return np.dot(np.dot(u, smat), vt)


def grid_score(index_i, index_j, grid_size_x, grid_size_y, grids, type='std'):
    # default was std
    if type == 'std':
        return np.std(grids[index_i * grid_size_y:(index_i + 1) * (grid_size_y),
         index_j * grid_size_x:(index_j + 1) * (grid_size_x)])
    elif type == 'mean':
        return np.mean(grids[index_i * grid_size_y:(index_i + 1) * (grid_size_y),
         index_j * grid_size_x:(index_j + 1) *
                                                                                                        (grid_size_x)])
    else:
        return np.amax(grids[index_i * grid_size_y:(index_i + 1) *
         (grid_size_y), index_j * grid_size_x:(index_j + 1) *
                                                                                                        (grid_size_x)])


def get_row_col_index(grid_index, grid_size_x, grid_size_y, grids_2d):
    num_grids_row = grids_2d.shape[1] // grid_size_x
    row_index = grid_index // num_grids_row
    col_index = grid_index % num_grids_row
    return row_index, col_index


def get_pixels_from_grid_index(grid_index, grid_size_x, grid_size_y, grids_2d, imgArray):
    row_index, col_index = get_row_col_index(grid_index, grid_size_x, grid_size_y, grids_2d)
    last_row_ind = math.floor(imgArray.shape[0] / grid_size_y)
    y_residue = imgArray.shape[0] % grid_size_y
    if (row_index == last_row_ind) and y_residue != 0:
        y1 = row_index * grid_size_y + y_residue
    else:
        y1 = (row_index + 1) * grid_size_y

    last_col_ind = math.floor(imgArray.shape[1] / grid_size_x)
    x_residue = imgArray.shape[1] % grid_size_x

    if (col_index == last_col_ind) and x_residue != 0:
        x1 = col_index * grid_size_x + x_residue
    else:
        x1 = (col_index + 1) * grid_size_x

    return row_index * grid_size_y, y1, col_index * grid_size_x, x1


def gridify_and_add_margin(imgArray, grid_size_x, grid_size_y):
    height, width = imgArray.shape[:2]
    margined_x = np.copy(imgArray)
    padding_row = np.ones((height, grid_size_x - (width % grid_size_x)), dtype=np.float64) * np.mean(imgArray)
    if width % grid_size_x != 0:
        margined_x = np.hstack((imgArray, padding_row))
    padding_col = np.ones((grid_size_y - (height % grid_size_y), margined_x.shape[1]), dtype=np.float64) * np.mean(imgArray)
    margined_image = np.copy(margined_x)
    if height % grid_size_y != 0:
        # Append the white row to each row in the image
        margined_image = np.vstack((margined_x, padding_col))
    num_grids_x = margined_image.shape[1] // grid_size_x
    num_grids_y = margined_image.shape[0] // grid_size_y

    return margined_image.reshape((num_grids_y, grid_size_y, num_grids_x, grid_size_x))


def remove_margin(imgArray, grids):
    return grids[:imgArray.shape[0], :imgArray.shape[1]]


def visualize_grids(grids, list_of_large_grid_indices, grid_size_x, grid_size_y, imgArray):
    grids_2d = grids.reshape(grids.shape[0] * grids.shape[1], grids.shape[2] * grids.shape[3])
    grids_vis = imgArray.copy()
    for index in range(grids.shape[0] * grids.shape[2]):
        y_0, y_1, x_0, x_1 = get_pixels_from_grid_index(index, grid_size_x, grid_size_y, grids_2d, imgArray)
        if index in list_of_large_grid_indices:
            # yellow ones are the complex ones
            grids_vis[y_0:y_1, x_0:x_1] = np.ones((y_1 - y_0, x_1 - x_0)) * 128
        else:
            # purple ones are the easier ones
            grids_vis[y_0:y_1, x_0:x_1] = np.ones((y_1 - y_0, x_1 - x_0)) * 256

    return grids_vis


def calculate_compression(src_img_size, dst_img_size):
    return (src_img_size - dst_img_size) / src_img_size


class SVD:
    def __init__(self, target_compression):
        self.target_compression = target_compression


    def _svd_compression(self, img):
        self.k = int((1 - self.target_compression) * (img.shape[0] * img.shape[1]) / (img.shape[0] + img.shape[1] + 1))
        u, s, vt = np.linalg.svd(img, full_matrices=False)
        smat = np.zeros(np.shape(img))
        smat[:len(s), :len(s)] = np.diag(s)
        svd = multiply_first_k(self.k, u, smat, vt).astype(np.uint8)
        space_required = self.k * (img.shape[0] + img.shape[1] + 1)
        return svd, space_required

    def __call__(self, img):
        if len(img.shape) == 2:
            svd, space_required = self._svd_compression(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_r, img_g, img_b = extract_rgb(img)
            compressed_r, space_required_r = self._svd_compression(img_r)
            compressed_g, space_required_g = self._svd_compression(img_g)
            compressed_b, space_required_b = self._svd_compression(img_b)
            svd = assemble_rgb(compressed_r, compressed_g, compressed_b)
            space_required = space_required_r + space_required_g + space_required_b
        return svd, space_required


def cropper(img_path, x1, y1, x2, y2):
    img = cv2.imread(img_path)
    # if img.shape[2] == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('.'.join(img_path.split('.')[:-1]) + "_cropped.png", img[y1:y2, x1:x2])
    print(f"wrote to {'.'.join(img_path.split('.')[:-1]) + '_cropped.png'}")