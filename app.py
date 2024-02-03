import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import metrics.globals
from metrics import core

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Porcess_params(BaseModel):
    processors: list
    target_path: str
    swapped_path: str
    source_path: str
    execution_threads: int


@app.post("/metrics")
def get_scores(params: Porcess_params):
    metrics.globals.processors = params.processors
    metrics.globals.source_path = params.source_path
    metrics.globals.target_path = params.target_path
    metrics.globals.swapped_path = params.swapped_path
    result_path = core.run()

    with open(result_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data
