from pydantic import BaseModel, Field
from typing import Literal

ServerMode = Literal["align", "heterogeneous"]
WeightMode = Literal["average"]


class DetectRequest(BaseModel):
    task_id: int = Field(alias="taskId")
    resource_url: str = Field(alias="resourceUrl")


class DetectResult(BaseModel):
    task_id: int = Field(alias="taskId")
    result: float
