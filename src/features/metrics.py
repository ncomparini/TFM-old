from datetime import datetime
from operator import itemgetter
from scipy.spatial.distance import cdist

from src.data.Dataset import Dataset
from src.features.utils.image import extract_points
import numpy as np
import os
import matplotlib.pyplot as plt

from src.models.log.LogTracker import LogTracker


def compute_distance_between_lists(actual_points, predicted_points):
    bb_up_left = np.asarray([min(actual_points, key=itemgetter(0))[0], min(actual_points, key=itemgetter(1))[1]])
    bb_down_right = np.asarray([max(actual_points, key=itemgetter(0))[0], max(actual_points, key=itemgetter(1))[1]])
    distance_bb = np.linalg.norm(bb_up_left - bb_down_right)

    s1 = np.asarray(actual_points)
    s2 = np.asarray(predicted_points)

    if len(s2) > 0:
        distances_normalized = np.min(cdist(s1, s2), axis=1) / distance_bb
        mean_distance = np.mean(distances_normalized)

        return mean_distance
    else:
        return None


def get_predicted_points(model, sample_image, is_image_output, threshold, number_of_landmarks=68):
    prediction = model.predict(sample_image)[0]
    if is_image_output:
        predicted_image = prediction[..., -1]
        return extract_points(predicted_image, threshold)
    else:
        predicted_points = prediction.astype(np.int8)
        return np.reshape(predicted_points, (number_of_landmarks, 2))


def compute_nppe_values(model, dataset: Dataset, dictionary, threshold, is_image_output, number_of_landmarks):
    nppe_values = []
    empty_values = 0

    for idx, (sample_image, _) in enumerate(dataset.dataset):
        predicted_points = get_predicted_points(model, sample_image, is_image_output, threshold, number_of_landmarks)

        image_name = os.path.basename(dataset.sample_images_urls[idx])
        actual_points = dictionary.get(image_name)

        # nppe: Normalized Point-to-Point Error
        nppe = compute_distance_between_lists(actual_points, predicted_points)
        if nppe is None:
            empty_values = empty_values + 1
        else:
            nppe_values.append(nppe)

    if empty_values != 0:
        print(f"WARNING. Found {empty_values} empty lists when computing NPPE.")

    return nppe_values


def get_image_proportion(nppe_values, limit_nppe):
    values_under_limit = np.argwhere(nppe_values < limit_nppe)
    len_values_under_limit = len(values_under_limit)
    len_nppe_values = len(nppe_values)

    image_proportion = len_values_under_limit / len_nppe_values
    nppe = nppe_values[len_values_under_limit] if len_values_under_limit != len_nppe_values else nppe_values[-1]

    return image_proportion, nppe


def compute_auc_points(sorted_nppe_values, ratio=10):
    auc_points = [(0, 0)]
    step = (0.005 / ratio)
    max_nppe = sorted_nppe_values[-1]

    for nppe in np.arange(0.0, max_nppe, step):
        image_proportion, current_nppe = get_image_proportion(sorted_nppe_values, nppe)
        auc_points.append((current_nppe, image_proportion))

    return np.asarray(auc_points)


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


def compute_auc_metrics(model, dataset: Dataset, gt_dictionary, is_image_output, threshold=0.75, number_of_landmarks=68,
                        area_limit=0.05, ratio=10):
    nppes = compute_nppe_values(model, dataset, gt_dictionary, threshold, is_image_output, number_of_landmarks)
    nppe_max, nppe_min, nppe_std, nppe_mean = np.max(nppes), np.min(nppes), np.std(nppes), np.mean(nppes)
    auc_points = compute_auc_points(np.sort(nppes), ratio=ratio)
    xs, ys = auc_points[:, 0], auc_points[:, 1]
    auc_plot = get_auc_plot(xs, ys, step=ratio, area_limit=area_limit)

    idx_last_valid_point = np.where(xs > area_limit)[-1][0]
    auc = np.trapz(ys[:idx_last_valid_point], xs[:idx_last_valid_point]) / area_limit

    metrics_dict = {
        "scalar-metrics": {
            "auc": auc,
            "nppe-max": nppe_max,
            "nppe-min": nppe_min,
            "nppe-std": nppe_std,
            "nppe-mean": nppe_mean,
        },
        "plot-metrics": {
            "auc-plot": auc_plot
        }
    }

    return metrics_dict


def save_eval_metrics(log_tracker: LogTracker, metrics_dict: dict, set_name: str):
    scalar_metrics = metrics_dict.get("scalar-metrics", {})
    plot_metrics = metrics_dict.get("plot-metrics", {})

    for metric_name, metric_value in scalar_metrics.items():
        path = f"evaluation/{set_name}/{metric_name}"
        log_tracker.save_static_variable(path, metric_value)

    for metric_name, metric_value in plot_metrics.items():
        path = f"evaluation/{set_name}/{metric_name}"
        log_tracker.save_image(path, metric_value)


def compute_and_save_metrics(model, dataset: Dataset, gt_dictionary, log_tracker: LogTracker, is_image_output: bool,
                             threshold: float = 0.75):
    start_time = datetime.now()
    metrics_dict = compute_auc_metrics(model, dataset, gt_dictionary, is_image_output, threshold)
    end_time = datetime.now()

    save_eval_metrics(log_tracker, metrics_dict, dataset.tag)

    auc = metrics_dict.get("scalar-metrics", {}).get("auc", None)
    print(f"Evaluation {dataset.tag} set finished. Elapsed time: {end_time - start_time}. AUC: {auc}")
