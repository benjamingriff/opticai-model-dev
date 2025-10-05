from abc import ABC, abstractmethod


class AnnotationExporter(ABC):
    @abstractmethod
    def export_dataset(self, dataset, output_dir, dataset_name):
        pass

    @abstractmethod
    def export_predictions(self, video_path, segments, output_dir):
        pass


class AnnotationImporter(ABC):
    @abstractmethod
    def import_project(self, project_dir):
        pass
