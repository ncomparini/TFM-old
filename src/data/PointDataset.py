import os

from src.data.Dataset import Dataset
from src.features.utils.image import normalize
from src.features.utils.io import read_image_urls, load_image


class PointDataset(Dataset):

    def __init__(self, path_sample_images, points_dictionary, batch_size, shuffle, tag, number_of_landmarks):
        super().__init__(batch_size, tag)
        self.path_sample_images = path_sample_images

        self.number_of_landmarks = number_of_landmarks

        self.sample_images_urls = read_image_urls(self.path_sample_images)
        self.target_points_list = self._create_points_list(points_dictionary)

        self.dataset = self._create_dataset(shuffle)

    def _load_image_and_points(self, input_img_file, output_points):
        input_img = load_image(input_img_file, img_type="jpg")
        norm_input_img = normalize(input_img)

        norm_output_points = output_points[:self.number_of_landmarks]

        return norm_input_img, norm_output_points

    def _create_points_list(self, points_dictionary):
        return [points_dictionary[os.path.basename(url)] for url in self.sample_images_urls]

    def _create_dataset(self, shuffle=True):
        tensors_list = (self.sample_images_urls, self.target_points_list)
        buffer_size = len(self.sample_images_urls)
        return self.create_dataset(tensors_list, self._load_image_and_points, buffer_size, shuffle)
