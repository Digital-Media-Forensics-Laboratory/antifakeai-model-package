from afk_toolset_http import AfkServer
from afk_toolset_http import DetectRequest, DetectResult
from typing import Annotated
from fastapi import FastAPI, Depends
from model_impl import get_server

app = FastAPI()


@app.post("/detect")
async def detect(
    request: DetectRequest, server: Annotated[AfkServer, Depends(get_server)]
) -> DetectResult:
    result = server.perform_detect_single(request)
    return result
