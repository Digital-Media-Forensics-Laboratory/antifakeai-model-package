from .afk_model import AfkModel
from .afk_data_model import DetectRequest, DetectResult, ServerMode, WeightMode
from .afk_detector import IAfkDetector, AfkDetector
from typing import List, Optional, Any, Literal


class AfkServer:

    def __init__(
        self,
        name: str,
        server_mode: ServerMode = "align",
        weight_mode: WeightMode = "average",
    ):
        self.detector: IAfkDetector = AfkDetector(server_mode, weight_mode)
        self.name = name
        self.mode = server_mode

    def register_model(self, model: AfkModel):
        self.detector.append_model(model)

    def perform_detect_single(self, request: DetectRequest) -> DetectResult:
        return self.detector.detect_single(request)
