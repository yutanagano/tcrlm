import multiprocessing as mp
import pytest
from src import models
from src.models.embedder import _Embedder


mp.set_start_method("spawn")
