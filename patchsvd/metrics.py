from cv2 import PSNR as psnr
import numpy as np
from skimage.metrics import structural_similarity as ssim


def mse(src_img, dst_img):
    if np.shape(src_img) == np.shape(dst_img):
        err = np.sum((src_img.astype("float") - dst_img.astype("float")) ** 2)
        err /= float(src_img.shape[0] * src_img.shape[1])
        return err
    return None



class ExperimentMetrics:
    def __init__(self, total):
        self.sum_ssim = 0
        self.sum_psnr = 0
        self.sum_mse = 0
        self.total = total

    def update_sum_ssim(self, new_value):
        self.sum_ssim += new_value

    def update_sum_psnr(self, new_value):
        self.sum_psnr += new_value

    def update_sum_mse(self, new_value):
        self.sum_mse += new_value

    def get_avg_ssim(self):
        return self.sum_ssim/self.total

    def get_avg_psnr(self):
        return self.sum_psnr/self.total

    def get_avg_mse(self):
        return self.sum_mse/self.total


class SampleMetrics:
    def __init__(self, experiment_metrics: ExperimentMetrics):
        self.ssim = 0
        self.psnr = 0
        self.mse = 0
        self.experiment_metrics = experiment_metrics

    def compute_metrics(self, transformed, reference):
        if len(transformed.shape) == 2:
            channel_axis = None
        else:
            channel_axis = 2
        self.ssim = ssim(reference, transformed, channel_axis=channel_axis)
        self.psnr = psnr(reference, transformed)
        self.mse = mse(reference, transformed)
        self.update_experiment_metrics()

    def update_experiment_metrics(self):
        self.experiment_metrics.update_sum_ssim(self.ssim)
        self.experiment_metrics.update_sum_psnr(self.psnr)
        self.experiment_metrics.update_sum_mse(self.mse)
