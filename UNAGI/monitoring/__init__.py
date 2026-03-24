from importlib import import_module

__all__ = ["JobMonitorService", "create_monitor_app", "generate_analysis_visualizations", "export_analysis_share_bundle"]


def __getattr__(name):
    if name == "JobMonitorService":
        return import_module(".service", __name__).JobMonitorService
    if name == "create_monitor_app":
        return import_module(".webapp", __name__).create_monitor_app
    if name == "generate_analysis_visualizations":
        return import_module(".visualization_pipeline", __name__).generate_analysis_visualizations
    if name == "export_analysis_share_bundle":
        return import_module(".share_bundle", __name__).export_analysis_share_bundle
    raise AttributeError(name)
