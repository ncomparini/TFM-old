import neptune.new as neptune
from neptune.new.types import File


class LogTracker:
    def __init__(self, config, run_id=None):
        self.project_name = config["project-name"]
        self.api_token = config["api-token"]
        self.name = config["run-name"]
        self.manager = self._wrapped_create_manager(self.name, run_id)
        self.run_id = run_id if run_id is not None else self.get_run_id()

    def _wrapped_create_manager(self, name, run_id):
        if run_id is None:
            return self._create_manager(name)
        else:
            return self._fetch_existing_run(name, run_id)

    def _fetch_existing_run(self, name, run_id):
        return neptune.init(project=self.project_name, api_token=self.api_token, name=name, run=run_id)

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

    def save_image(self, key, image):
        self.manager[key].upload(File.as_image(image))

    def get_field_value(self, key):
        return self.manager[key].fetch()

    def get_run_id(self):
        return self.get_field_value("sys/id")
