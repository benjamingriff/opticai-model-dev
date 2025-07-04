import importlib


def get_dataset(name: str, task: str, scope: str):
    """
    Dynamically imports the correct dataset class.
    Args:
        name: dataset name (e.g., "cataract21")
        task: task type (e.g., "phase_segmentation")
        scope: "all", "single", or "unlabelled"
    Returns:
        Dataset class
    """
    module_path = f"datasets.{name.replace('-', '')}.{task}.dataset_{scope}"
    class_name = f"{name.replace('-', '').capitalize()}{''.join(word.capitalize() for word in task.replace('_', ' ').split())}Dataset{scope.capitalize()}"

    try:
        module = importlib.import_module(module_path)
        DatasetClass = getattr(module, class_name)
        return DatasetClass
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(
            f"Could not find dataset class '{class_name}' in module '{module_path}'."
        ) from e
