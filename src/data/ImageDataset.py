from src.data.Dataset import Dataset
from src.features.utils.image import normalize
from src.features.utils.io import read_image_urls, load_image


class ImageDataset(Dataset):

    def __init__(self, path_sample_images, path_target_images, batch_size, shuffle, tag):
        super().__init__(batch_size, tag)
        self.path_sample_images = path_sample_images
        self.path_target_images = path_target_images

        self.sample_images_urls = read_image_urls(self.path_sample_images)
        self.target_images_urls = read_image_urls(self.path_target_images)

        self.dataset = self._create_dataset_wrapper(shuffle)

    def _load_paired_image(self, input_img_file, output_img_file):
        input_img = load_image(input_img_file, img_type="jpg")
        norm_input_img = normalize(input_img)

        output_img = load_image(output_img_file, img_type="png")
        norm_output_img = normalize(output_img)

        return norm_input_img, norm_output_img

    def _create_dataset_wrapper(self, shuffle=True):
        tensors_list = (self.sample_images_urls, self.target_images_urls)
        buffer_size = len(self.sample_images_urls)
        return self.create_dataset(tensors_list, self._load_paired_image, buffer_size, shuffle)
