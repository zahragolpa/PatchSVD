import os
import csv
import re
import matplotlib.pyplot as plt
import numpy as np
from utils import cropper

# Path to the "output" directory
output_dir = "output"  # Replace with your actual path

# Function to extract hyperparameter and compression ratio from the filename
def extract_hyperparam_and_ratio(file_name, dataset_name):
    match = re.match(f"dataset_{dataset_name}_"+r"P_x_(\d+)_P_y_(\d+)_target_compression_(\d+\.\d+).csv", file_name)
    if match:
        hyperparameter = int(match.group(1))
        compression_ratio = float(match.group(3))
        return hyperparameter, compression_ratio
    else:
        return None, None


def extract_mse_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)
        if len(rows) > 1 and len(rows[1]) > 1:
            # if abs(float(rows[1][5]) - float(rows[2][5])) > 0.05:
            #     return None, None, None
            return float(rows[1][2]), float(rows[2][2]), float(rows[3][2])  # Assuming SSIM is at the second row, second column


def extract_ssim_score_f_from_csv(file_path):
    return extract_ssim_from_csv(file_path.replace("SCORETYPE", ".csv"))[0],\
        extract_ssim_from_csv(file_path.replace("SCORETYPE", "_Score_Mean.csv"))[0],\
            extract_ssim_from_csv(file_path.replace("SCORETYPE", "_Score_Max.csv"))[0]

def extract_psnr_score_f_from_csv(file_path):
    return extract_psnr_from_csv(file_path.replace("SCORETYPE", ".csv"))[0],\
        extract_psnr_from_csv(file_path.replace("SCORETYPE", "_Score_Mean.csv"))[0],\
            extract_psnr_from_csv(file_path.replace("SCORETYPE", "_Score_Max.csv"))[0]

def extract_mse_score_f_from_csv(file_path):
    return extract_mse_from_csv(file_path.replace("SCORETYPE", ".csv"))[0],\
        extract_mse_from_csv(file_path.replace("SCORETYPE", "_Score_Mean.csv"))[0],\
            extract_mse_from_csv(file_path.replace("SCORETYPE", "_Score_Max.csv"))[0]


# Function to extract SSIM values from CSV files
def extract_ssim_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)
        if len(rows) > 1 and len(rows[1]) > 1:
            return float(rows[1][1]), float(rows[2][1]), float(rows[3][1])  # Assuming SSIM is at the second row, second column


def extract_psnr_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)
        if len(rows) > 1 and len(rows[1]) > 1:
            return float(rows[1][3]), float(rows[2][3]), float(rows[3][3])  # Assuming SSIM is at the second row, second column


def plot_metric_vs_compression_ratios_for_p_sizes(hyperparameter_values, compression_ratios, dataset_name, metric_name):
    for hyperparameter in hyperparameter_values:
        metric_patch_svd = []
        metric_jpeg = []
        metric_svd = []
        # if hyperparameter_values.index(hyperparameter) % 2 == 0:
        #     continue

        # Iterate through compression ratios
        for compression_ratio in compression_ratios:
            file_name = f"dataset_{dataset_name}_P_x_{hyperparameter}_P_y_{hyperparameter}_target_compression_{compression_ratio}.csv"
            file_path = os.path.join(output_dir, file_name)

            if os.path.exists(file_path):
                if metric_name == "SSIM":
                    metric_value_patch_svd, metric_value_jpeg, metric_value_svd = extract_ssim_from_csv(file_path)
                elif metric_name == "PSNR":
                    metric_value_patch_svd, metric_value_jpeg, metric_value_svd = extract_psnr_from_csv(file_path)
                elif metric_name == "MSE":
                    metric_value_patch_svd, metric_value_jpeg, metric_value_svd = extract_mse_from_csv(file_path)
                
                metric_patch_svd.append(metric_value_patch_svd)
                # metric_jpeg.append(metric_value_jpeg)
                # metric_svd.append(metric_value_svd)

        plt.plot(compression_ratios[:len(metric_patch_svd)], metric_patch_svd, label=f'PatchSVD - Patch size = {hyperparameter}')
        # plt.plot(compression_ratios[:len(metric_patch_svd)], metric_jpeg, '-.', label=f'JPEG')
        # plt.plot(compression_ratios[:len(metric_patch_svd)], metric_svd, '--', label=f'SVD')
        plt.xlabel('Compression Ratio')
        plt.ylabel(f'{metric_name}')
        plt.title(f'{metric_name} vs Compression Ratio for Different Patch Sizes')
        plt.legend()
        plt.autoscale()
        # plt.ylim((0.75, 0.85))
        plt.savefig(f'patchsvd_various_patch_sizes_{dataset_name}_{metric_name}.png')


def metric_vs_compression_ratio_different_methods(hyperparameter, compression_ratios, dataset_name, metric_name):

    metric_patch_svd = []
    metric_jpeg = []
    metric_svd = []

    # Iterate through compression ratios
    for compression_ratio in compression_ratios:
        file_name = f"dataset_{dataset_name}_P_x_{hyperparameter}_P_y_{hyperparameter}_target_compression_{compression_ratio}.csv"
        file_path = os.path.join(output_dir, file_name)

        if os.path.exists(file_path):
            if metric_name == "SSIM":
                metric_value_patch_svd, metric_value_jpeg, metric_value_svd = extract_ssim_from_csv(file_path)
            elif metric_name == "PSNR":
                metric_value_patch_svd, metric_value_jpeg, metric_value_svd = extract_psnr_from_csv(file_path)
            elif metric_name == "MSE":
                metric_value_patch_svd, metric_value_jpeg, metric_value_svd = extract_mse_from_csv(file_path)
            
            metric_patch_svd.append(metric_value_patch_svd)
            metric_jpeg.append(metric_value_jpeg)
            metric_svd.append(metric_value_svd)
    # Plotting with non-empty SSIM scores
    print('Metric')
    print(metric_name)
    print('dataset')
    print(dataset_name)
    print('Compression ratios')
    print(f'{compression_ratios}')
    print('PatchSVD')
    print([float(f"{value:.2f}") for value in metric_patch_svd])
    print('JPEG')
    print([float(f"{value:.2f}") for value in metric_jpeg])
    print('SVD')
    print([float(f"{value:.2f}") for value in metric_svd])

    plt.plot(compression_ratios[:len(metric_patch_svd)], metric_patch_svd, label=f'PatchSVD - Patch size = {hyperparameter}', color='green')
    plt.plot(compression_ratios[:len(metric_patch_svd)], metric_jpeg, label=f'JPEG', color='blue')
    plt.plot(compression_ratios[:len(metric_patch_svd)], metric_svd, label=f'SVD', color='red')
    plt.xlabel('Compression Ratio')
    plt.ylabel(f'{metric_name}')
    plt.title(f'{metric_name} vs Compression Ratio for Different methods')
    plt.legend()
    plt.autoscale()
    plt.savefig(f'patchsvd_comparison_with_jpeg_and_svd_for_p_{hyperparameter}_{dataset_name}_{metric}.png')


def compare_score_functions_wrt_metric(hyperparameter, compression_ratios, dataset_name, metric_name):

    metric_patch_svd_std = []
    metric_patch_svd_mean = []
    metric_patch_svd_max = []

    for compression_ratio in compression_ratios:
        file_name = f"dataset_{dataset_name}_P_x_{hyperparameter}_P_y_{hyperparameter}_target_compression_{compression_ratio}SCORETYPE"
        file_path = os.path.join(output_dir, file_name)

        # if os.path.exists(file_path):
        if metric_name == "SSIM":
            metric_value_patch_svd_std, metric_value_patch_svd_mean, metric_value_patch_svd_max = extract_ssim_score_f_from_csv(file_path)
        elif metric_name == "PSNR":
            metric_value_patch_svd_std, metric_value_patch_svd_mean, metric_value_patch_svd_max = extract_psnr_score_f_from_csv(file_path)
        elif metric_name == "MSE":
            metric_value_patch_svd_std, metric_value_patch_svd_mean, metric_value_patch_svd_max = extract_mse_score_f_from_csv(file_path)
        
        metric_patch_svd_std.append(metric_value_patch_svd_std)
        metric_patch_svd_mean.append(metric_value_patch_svd_mean)
        metric_patch_svd_max.append(metric_value_patch_svd_max)

    plt.plot(compression_ratios[:len(metric_patch_svd_std)], metric_patch_svd_std, label=f'PatchSVD - std', color='green')
    plt.plot(compression_ratios[:len(metric_patch_svd_mean)], metric_patch_svd_mean, label=f'PatchSVD - mean', color='blue')
    plt.plot(compression_ratios[:len(metric_patch_svd_max)], metric_patch_svd_max, label=f'PatchSVD - max', color='red')
    plt.xlabel('Compression Ratio')
    plt.ylabel(f'{metric_name}')
    plt.title(f'{metric_name} vs Compression Ratio for Different Patch Sizes')
    plt.legend()
    plt.autoscale()
    # plt.ylim((0.75, 0.85))
    plt.savefig(f'compare_score_functions_wrt_metric_{dataset_name}_{metric_name}.png')



# Sample data
hyperparameter_values = set()
compression_ratios = set()
dataset_name = "Kodak"
metric = "PSNR"

# Iterate through CSV files in the "output" directory
for file_name in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".csv"):
        hyperparameter, compression_ratio = extract_hyperparam_and_ratio(file_name, dataset_name)
        if hyperparameter is not None and compression_ratio is not None:
            hyperparameter_values.add(hyperparameter)
            compression_ratios.add(compression_ratio)

hyperparameter_values = sorted(list(hyperparameter_values))
compression_ratios = sorted(list(compression_ratios))

# metric_vs_compression_ratio_different_methods(10, compression_ratios, dataset_name, metric)
# plt.cla()
# plot_metric_vs_compression_ratios_for_p_sizes(hyperparameter_values, compression_ratios, dataset_name, metric)
compare_score_functions_wrt_metric(16, compression_ratios, dataset_name, metric)
# cropper("output/dataset_Kodak_P_x_8_P_y_8_target_compression_0.2/8_svd.png", 378, 368, 456, 446)
# cropper("output/dataset_Kodak_P_x_32_P_y_32_target_compression_0.15/5_svd.png", 358, 232, 452, 309)
# cropper("output/dataset_Kodak_P_x_32_P_y_32_target_compression_0.85/5_jpeg.png", 282, 346, 390, 378)
