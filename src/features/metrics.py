from operator import itemgetter
from scipy.spatial.distance import cdist
from src.data.ImageDataset import ImageDataset
from src.features.utils.image import extract_points
import numpy as np
import os
import matplotlib.pyplot as plt


def compute_distance_between_lists(actual_points, predicted_points):
    bb_up_left = np.asarray([min(actual_points, key=itemgetter(0))[0], min(actual_points, key=itemgetter(1))[1]])
    bb_down_right = np.asarray([max(actual_points, key=itemgetter(0))[0], max(actual_points, key=itemgetter(1))[1]])
    distance_bb = np.linalg.norm(bb_up_left - bb_down_right)

    s1 = np.asarray(actual_points)
    s2 = np.asarray(predicted_points)
    distances_normalized = np.min(cdist(s1, s2), axis=1) / distance_bb
    mean_distance = np.mean(distances_normalized)

    return mean_distance


def compute_nppe_values(model, dataset: ImageDataset, dictionary):
    nppe_values = []

    for idx, (sample_image, target_image) in enumerate(dataset.dataset):
        predicted_image = model.predict(sample_image)[0][..., -1]
        image_name = os.path.basename(dataset.sample_images_urls[idx])

        actual_points = dictionary.get(image_name)
        predicted_points = extract_points(predicted_image, threshold=0.75)

        # nppe: Normalized Point-to-Point Error
        nppe = compute_distance_between_lists(actual_points, predicted_points)

        nppe_values.append(nppe)

    return nppe_values


def get_image_proportion(auc_values, limit_nppe):
    image_proportion = -1
    nppe = -1
    for i, current_nppe in enumerate(auc_values):
        if current_nppe >= limit_nppe:
            image_proportion = i / len(auc_values)
            nppe = current_nppe
            break

    return image_proportion, nppe


def compute_auc_points(sorted_auc_values, ratio=10):
    auc_points = [(0, 0)]
    step = (0.005 / ratio)
    max_nppe = sorted_auc_values[-1]

    for nppe in np.arange(0.0, max_nppe, step):
        image_proportion, current_nppe = get_image_proportion(sorted_auc_values, nppe)
        auc_points.append((current_nppe, image_proportion))

    return np.asarray(auc_points)


def compute_auc_metrics(model, dataset: ImageDataset, dictionary, area_limit=0.05, ratio=10):
    nppes = compute_nppe_values(model, dataset, dictionary)
    nppe_max, nppe_min, nppe_std, nppe_mean = np.max(nppes), np.min(nppes), np.std(nppes), np.mean(nppes)
    auc_points = compute_auc_points(np.sort(nppes), ratio=ratio)
    xs, ys = auc_points[:, 0], auc_points[:, 1]
    auc_plot = get_auc_plot(xs, ys, step=ratio, area_limit=area_limit)

    idx_last_valid_point = np.where(xs > area_limit)[-1][0]
    auc = np.trapz(ys[:idx_last_valid_point], xs[:idx_last_valid_point]) / area_limit

    return auc, auc_plot, nppe_max, nppe_min, nppe_std, nppe_mean


def get_auc_plot(xs, ys, step=10, area_limit=0.05):
    figure, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.scatter(xs[::step], ys[::step], marker='s', color='goldenrod')
    ax.plot(xs, ys, color='darkorange')
    ax.hlines(np.arange(0.0, 1.0, 0.1), xmin=0.0, xmax=0.05, linestyles='dashed', colors='darkgrey')
    ax.vlines(np.arange(0.0, 0.05, 0.01), ymin=0.0, ymax=1.0, linestyles='dashed', colors='darkgrey')
    ax.set_xlabel("Normalized Point-to-Point error")
    ax.set_ylabel("Images Proportion")
    ax.set_xlim(0.0, area_limit)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))

    return figure