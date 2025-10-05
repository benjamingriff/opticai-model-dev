from annotations.supervisely.export import SuperviselyExporter
from annotations.supervisely.load import SuperviselyImporter


def get_exporter(name: str):
    if name == "supervisely":
        return SuperviselyExporter()
    raise NotImplementedError()


def get_importer(name: str):
    if name == "supervisely":
        return SuperviselyImporter()
    raise NotImplementedError()
