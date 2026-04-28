from app.src.repositories import BaseModelRegistry
from app.src.utils import VideoReader


class Classifier:
    def __init__(self, model_registry: BaseModelRegistry, reader: VideoReader):
        self.model_registry = model_registry
        self.reader = reader
