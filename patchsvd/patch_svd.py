import numpy as np
import os
import cv2
from PIL import Image
from utils import multiply_first_k, gridify_and_add_margin, grid_score, get_pixels_from_grid_index, remove_margin, \
    visualize_grids, extract_rgb, assemble_rgb, SVD


def calculate_svd_diff(input_image_array):
    u, s, vt = np.linalg.svd(input_image_array)
    smat = np.zeros(np.shape(input_image_array))
    smat[:len(s), :len(s)] = np.diag(s)
    truncated_k = 1
    truncated = multiply_first_k(truncated_k, u, smat, vt)

    input_image_array = np.array(input_image_array).astype(np.float64)

    diff = np.abs(np.subtract(input_image_array, truncated))
    return diff


def populate_score_dict(diff_grids, grid_size_x, grid_size_y):
    index_score_dict = {}
    ind = 0
    diff_grids_2d = diff_grids.reshape(diff_grids.shape[0] * diff_grids.shape[1], diff_grids.shape[2] * diff_grids.shape[3])

    for index_i in range(diff_grids.shape[0]):
        for index_j in range(diff_grids.shape[2]):
            index_score_dict[ind] = grid_score(index_i, index_j, grid_size_x, grid_size_y, diff_grids_2d)
            ind += 1
    return index_score_dict


def get_truncated_grids(input_image_array, grids, grid_size_x, grid_size_y, complex_grids_index_list, k_complex, k_smooth):
    truncated_grids_2d = grids.reshape(grids.shape[0] * grids.shape[1], grids.shape[2] * grids.shape[3])

    space_required = 0
    for index in range(grids.shape[0] * grids.shape[2]):
        y_0, y_1, x_0, x_1 = get_pixels_from_grid_index(index, grid_size_x, grid_size_y, truncated_grids_2d, input_image_array)
        u, s, vt = np.linalg.svd(truncated_grids_2d[y_0:y_1, x_0:x_1], full_matrices=False)
        smat = np.zeros(np.shape(truncated_grids_2d[y_0:y_1, x_0:x_1]))
        smat[:len(s), :len(s)] = np.diag(s)

        if index in complex_grids_index_list:
            k = k_complex
        else:
            k = k_smooth
        truncated = multiply_first_k(k, u, smat, vt)
        space_required += k * (np.shape(u)[0] + np.shape(vt)[1] + 1)
        truncated_grids_2d[y_0:y_1, x_0:x_1] = truncated
    return truncated_grids_2d, space_required


def calc_complex_grids(grids, diff_grids, grid_size_x, grid_size_y, compression_ratio, k_complex, k_smooth):
    index_score_dict = populate_score_dict(diff_grids, grid_size_x, grid_size_y)
    largest_grids = sorted(index_score_dict.items(), key=lambda item: item[1], reverse=True)

    proportion_of_complex = ((grid_size_x * grid_size_y * (1 - compression_ratio)) /
                             (grid_size_x + grid_size_y + 1) - k_smooth) / (k_complex - k_smooth)
    # print(f"proportion of complex is {proportion_of_complex} and number of complex is \
    #     {proportion_of_complex * grids.shape[0] * grids.shape[2]}")
    if proportion_of_complex * grids.shape[0] * grids.shape[2] < 1:
        # fallback to SVD
        # print("Fallback to SVD")
        return None
    n_complex_grids = max(int(proportion_of_complex * grids.shape[0] * grids.shape[2]), 1)
    # print(f'{n_complex_grids} out of {grids.shape[0] * grids.shape[2]} grids are complex.')

    complex_grids_index_list = [grid[0] for grid in largest_grids[:min(n_complex_grids, len(largest_grids))]]
    return complex_grids_index_list


class PatchSVD:
    def __init__(self, grid_size_x, grid_size_y, compression_ratio, output_dir='output', visualize=False,
                 visualization_limit=10):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.compression_ratio = compression_ratio
        self.k_complex = min(self.grid_size_x, self.grid_size_y)
        self.k_smooth = max(int(self.k_complex // 5), 1)
        self.output_dir = output_dir
        self.visualize = visualize
        self.visualization_limit = visualization_limit
        self.already_visualized = 0

    def __call__(self, sample, name='temp'):
        img = np.array(sample)
        cv2.imwrite(os.path.join(self.output_dir, f"{name}_original.png"), img)
        img = cv2.imread(os.path.join(self.output_dir, f"{name}_original.png"), cv2.IMREAD_UNCHANGED)
        os.remove(os.path.join(self.output_dir, f"{name}_original.png"))
        fallback = False
        if len(img.shape) == 2:
            mode = 'L'
            compressed_image, list_of_large_grid_indices, space_required = self._patch_svd(img)
            if not (list_of_large_grid_indices):
                fallback = True
            else:
                grids = gridify_and_add_margin(img, grid_size_x=self.grid_size_x, grid_size_y=self.grid_size_y)
                if self.visualize and self.already_visualized < self.visualization_limit:
                    vis_image = visualize_grids(grids, list_of_large_grid_indices, self.grid_size_x, self.grid_size_y, img)
                    cv2.imwrite(os.path.join(self.output_dir, f"{name}_visualize.png"), vis_image)
                    self.already_visualized += 1
        else:
            mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_r, img_g, img_b = extract_rgb(img)
            compressed_r, list_of_large_grid_indices, space_required_r = self._patch_svd(img_r)
            compressed_g, list_of_large_grid_indices, space_required_g = self._patch_svd(img_g)
            compressed_b, list_of_large_grid_indices, space_required_b = self._patch_svd(img_b)
            
            if not (list_of_large_grid_indices):
                fallback = True
            else:
                grids = gridify_and_add_margin(img_r, grid_size_x=self.grid_size_x, grid_size_y=self.grid_size_y)
                if self.visualize and self.already_visualized < self.visualization_limit:
                    vis_image = visualize_grids(grids, list_of_large_grid_indices, self.grid_size_x, self.grid_size_y, img_r)
                    cv2.imwrite(os.path.join(self.output_dir, f"{name}_visualize.png"), vis_image)
                    self.already_visualized += 1
                compressed_image = assemble_rgb(compressed_r, compressed_g, compressed_b)
                if self.visualize and self.already_visualized < self.visualization_limit:
                    cv2.imwrite(os.path.join(self.output_dir, f"{name}_patch_svd.png"), img)
                space_required = space_required_r + space_required_g + space_required_b
        if fallback:
            print("Fallback to SVD")
            svd = SVD(self.compression_ratio)
            return svd(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return Image.fromarray(compressed_image.astype('uint8'), mode), space_required

    def _patch_svd(self, input_image_array):
        if self.compression_ratio >= 1:
            return None

        diff = calculate_svd_diff(input_image_array)
        # cv2.imwrite("diff.png", diff)
        diff_grids = gridify_and_add_margin(diff, grid_size_x=self.grid_size_x, grid_size_y=self.grid_size_y)
        grids = gridify_and_add_margin(input_image_array, grid_size_x=self.grid_size_x, grid_size_y=self.grid_size_y)

        complex_grids_index_list = calc_complex_grids(grids, diff_grids, self.grid_size_x, self.grid_size_y,
                                                      self.compression_ratio, self.k_complex, self.k_smooth)
        if not complex_grids_index_list:
            return None, None, None
        truncated_grids_2d, space_required = get_truncated_grids(input_image_array, grids, self.grid_size_x, self.grid_size_y,
                                                                 complex_grids_index_list, self.k_complex, self.k_smooth)
        compressed = remove_margin(input_image_array, truncated_grids_2d).astype(np.uint8)
        return compressed, complex_grids_index_list, space_required

