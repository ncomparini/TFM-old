import neptune.new as neptune
from neptune.new.types import File


class LogTracker:
    def __init__(self, config, name):
        self.project_name = config["project-name"]
        self.api_token = config["api-token"]
        self.manager = self._create_manager(name)

    def _create_manager(self, name):
        return neptune.init(project=self.project_name, api_token=self.api_token, name=name)

    def save_static_variable(self, key, value):
        self.manager[key] = value

    def add_tags(self, tags):
        self.manager["sys/tags"].add(tags)

    def log_variable(self, key, value):
        self.manager[key].log(value)

    def log_image(self, key, tf_tensor):
        self.manager[key].log(File.as_image(tf_tensor))
