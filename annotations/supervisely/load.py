from annotations.base import AnnotationImporter


class SuperviselyImporter(AnnotationImporter):
    def import_project(self, project_dir):
        """
        Read Supervisely JSON files and convert to your internal format.
        Returns a list of (video_path, segments) or your Dataset subclass.
        """
        # Parse meta.json and ann/*.json
        # Align with videos/
        pass
