import argparse
import numpy as np
from torch.utils.data import random_split
from utils import to_jpeg, SVD, calculate_compression
import os
import cv2
import csv
from metrics import SampleMetrics, ExperimentMetrics
from patch_svd import PatchSVD

parser = argparse.ArgumentParser(description='Welcome to PatchSVD Experiments!')
parser.add_argument('--dataset', default=None, help='Name of the dataset. Choose from MNIST, CIFAR-10, FGVC_Aircraft, '
                                                       'EuroSAT, Kodak, and CLIC.', choices=['MNIST', 'CIFAR-10', 'FGVC_Aircraft', 'EuroSAT', 'Kodak', 'CLIC'])
parser.add_argument('--p_x', help='Patch size along the x-axis.', default=5, type=int)
parser.add_argument('--p_y', help='Patch size along the y-axis.', default=5, type=int)
parser.add_argument('--target-compression', '-c', help='Your target compression rate. Should be less than 1.',
                    default=0.1, type=float)
parser.add_argument('--output-dir', help='Path to saving dir.', default='output')
parser.add_argument('--visualize', help='whether visualize the score maps for PatchSVD or not.', default=False,
                    action='store_true')
parser.add_argument('--visualization_limit', help='how many images do you want to visualize', default=10, type=int)
parser.add_argument('--img-path', help='Path to image file you want to compress.', default=None)

def prep_the_dataset(args):
    if args.dataset == 'MNIST':
        from torchvision.datasets import MNIST
        dataset = MNIST(root='./', train=False)
    elif args.dataset == 'FGVC_Aircraft':
        from torchvision.datasets import FGVCAircraft
        dataset = FGVCAircraft(root='FGVC_Aircraft', split='test', download=False)
    elif args.dataset == 'EuroSAT':
        from torchvision.datasets import EuroSAT
        dataset = EuroSAT(root='./', download=False)
        dataset = random_split(dataset, [0.8, 0.2])[1]
    elif args.dataset == 'CIFAR-10':
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root='./', train=False, download=False)
    elif args.dataset == 'Kodak':
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root='Kodak')
    elif args.dataset == 'CLIC':
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root='CLIC')    
    elif not args.dataset:
        assert args.img_path, print("Neither a dataset nor an image file has been selected. Exiting...")
        return None
    else:
        return None
    return dataset


def main():
    args = parser.parse_args()
    dataset = prep_the_dataset(args)
    experiment_name = f'dataset_{args.dataset}_P_x_{args.p_x}_P_y_{args.p_y}_target_compression_{args.target_compression}'
    args.output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
    patch_svd_runner = PatchSVD(args.p_x, args.p_y, args.target_compression, args.output_dir,
                                args.visualize, args.visualization_limit)
    svd_runner = SVD(args.target_compression)
    if dataset:
        experiment_metrics_patch_svd = ExperimentMetrics(len(dataset))
        experiment_metrics_svd = ExperimentMetrics(len(dataset))
        experiment_metrics_jpeg = ExperimentMetrics(len(dataset))

        for sample_index, sample in enumerate(dataset):
            print(f"Running sample {sample_index} / {len(dataset)}")
            print(dataset.imgs[sample_index][0])
            img = np.array(sample[0])
            patch_svd_pil, patch_svd_space_required = patch_svd_runner(img, str(sample_index))
            patch_svd = np.array(patch_svd_pil)
            jpeg = to_jpeg(img, 1 - args.target_compression, args.output_dir)
            svd, svd_space_required = svd_runner(img)
            if sample_index < args.visualization_limit:
                cv2.imwrite(os.path.join(args.output_dir, f"{sample_index}_patch_svd.png"), patch_svd)
                cv2.imwrite(os.path.join(args.output_dir, f"{sample_index}_jpeg.png"), jpeg)
                cv2.imwrite(os.path.join(args.output_dir, f"{sample_index}_svd.png"), svd)

            sample_metrics_patch_svd = SampleMetrics(experiment_metrics_patch_svd)
            sample_metrics_svd = SampleMetrics(experiment_metrics_svd)
            sample_metrics_jpeg = SampleMetrics(experiment_metrics_jpeg)

            sample_metrics_patch_svd.compute_metrics(patch_svd, img)
            sample_metrics_svd.compute_metrics(svd, img)
            sample_metrics_jpeg.compute_metrics(jpeg, img)

            original_size = 1
            for shape in img.shape:
                original_size *= shape
            calculated_compression_ratio_svd = calculate_compression(original_size, svd_space_required)
            calculated_compression_ratio_patch_svd = calculate_compression(original_size, patch_svd_space_required)

            # print(f'compression ratio achieved by PatchSVD is {calculated_compression_ratio_patch_svd}')
            # print(f'compression ratio achieved by SVD is {calculated_compression_ratio_svd}')
        with open(args.output_dir + '.csv', 'w', newline='') as csv_file:
            experiment_writer = csv.writer(csv_file)
            experiment_writer.writerow(['Method Name', 'AVG SSIM', 'AVG MSE', 'AVG PSNR', 'Compression Rate'])
            experiment_writer.writerow(['PatchSVD', experiment_metrics_patch_svd.get_avg_ssim(),
                                        experiment_metrics_patch_svd.get_avg_mse(),
                                        experiment_metrics_patch_svd.get_avg_psnr(),
                                        calculated_compression_ratio_patch_svd])
            experiment_writer.writerow(['JPEG', experiment_metrics_jpeg.get_avg_ssim(),
                                    experiment_metrics_jpeg.get_avg_mse(),
                                    experiment_metrics_jpeg.get_avg_psnr(),
                                    args.target_compression])
            experiment_writer.writerow(['SVD', experiment_metrics_svd.get_avg_ssim(),
                                        experiment_metrics_svd.get_avg_mse(),
                                        experiment_metrics_svd.get_avg_psnr(),
                                        calculated_compression_ratio_svd])
    else:
        print(f"compressing image {args.img_path}")
        img = cv2.imread(args.img_path)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        experiment_metrics_patch_svd = ExperimentMetrics(1)
        experiment_metrics_svd = ExperimentMetrics(1)
        experiment_metrics_jpeg = ExperimentMetrics(1)
        patch_svd_pil, patch_svd_space_required = patch_svd_runner(img, args.img_path)
        patch_svd = np.array(patch_svd_pil)
        jpeg = to_jpeg(img, 1 - args.target_compression, args.output_dir)
        svd, svd_space_required = svd_runner(img)
        cv2.imwrite(os.path.join(args.output_dir, f"{args.img_path}_patch_svd.png"), patch_svd)
        cv2.imwrite(os.path.join(args.output_dir, f"{args.img_path}_jpeg.png"), jpeg)
        cv2.imwrite(os.path.join(args.output_dir, f"{args.img_path}_svd.png"), svd)

        sample_metrics_patch_svd = SampleMetrics(experiment_metrics_patch_svd)
        sample_metrics_svd = SampleMetrics(experiment_metrics_svd)
        sample_metrics_jpeg = SampleMetrics(experiment_metrics_jpeg)

        sample_metrics_patch_svd.compute_metrics(patch_svd, img)
        sample_metrics_svd.compute_metrics(svd, img)
        sample_metrics_jpeg.compute_metrics(jpeg, img)
        print(f"SSIM for PatchSVD, JPEG, SVD: \
            {experiment_metrics_patch_svd.get_avg_ssim(), experiment_metrics_jpeg.get_avg_ssim(), experiment_metrics_svd.get_avg_ssim()}")
        print(f"MSE for PatchSVD, JPEG, SVD: \
            {experiment_metrics_patch_svd.get_avg_mse(), experiment_metrics_jpeg.get_avg_mse(), experiment_metrics_svd.get_avg_mse()}")
        print(f"PSNR for PatchSVD, JPEG, SVD: \
            {experiment_metrics_patch_svd.get_avg_psnr(), experiment_metrics_jpeg.get_avg_psnr(), experiment_metrics_svd.get_avg_psnr()}")


if __name__ == "__main__":
    main()


