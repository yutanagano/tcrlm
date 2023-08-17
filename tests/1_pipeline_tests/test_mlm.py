from multiprocessing import Process
import pytest
import torch
import warnings

from src.model_trainer import ModelTrainer
from src.model.token_embedder import BetaCdrEmbedder
from src.model.self_attention_stack import SelfAttentionStackWithBuiltins
from src.model.mlm_token_prediction_projector import AminoAcidTokenProjector
from src.model.vector_representation_delegate import (
    AveragePoolVectorRepresentationDelegate,
)
from src.model.bert import Bert

import tests.resources.helper_functions as helper_functions


D_MODEL = 4
NUM_LAYERS = 2
N_HEAD = 2


def test_main(model_template, config, tmp_path):
    if not torch.cuda.is_available():
        warnings.warn("MLM test skipped due to hardware limitations")

    training_manager = ModelTrainer(config)
    expected_model_save_dir = tmp_path / "model_saves" / "foo_bar_baz_01"

    p = Process(
        target=training_manager.train,
        kwargs={"working_directory": tmp_path},
    )
    p.start()
    p.join()

    assert expected_model_save_dir.is_dir()
    assert helper_functions.model_saved(expected_model_save_dir, model_template)
    assert helper_functions.log_saved(
        expected_model_save_dir,
        expected_cols=[
            "epoch",
            "loss",
            "lr",
            "valid_loss",
            "valid_acc",
            "valid_top5_acc",
        ],
        expected_len=4,
    )
    assert helper_functions.config_saved(expected_model_save_dir, config)


@pytest.fixture
def model_template():
    token_embedder = BetaCdrEmbedder(embedding_dim=D_MODEL)
    self_attention_stack = SelfAttentionStackWithBuiltins(
        num_layers=NUM_LAYERS, d_model=D_MODEL, nhead=N_HEAD
    )
    mlm_token_prediction_projector = AminoAcidTokenProjector(d_model=D_MODEL)
    vector_embedding_delegate = AveragePoolVectorRepresentationDelegate(
        self_attention_stack
    )

    bert = Bert(
        token_embedder,
        self_attention_stack,
        mlm_token_prediction_projector,
        vector_embedding_delegate,
    )

    return bert


@pytest.fixture
def config():
    return {
        "training_delegate": {
            "class": "MlmTrainingDelegate",
            "initargs": {}
        },
        "model": {
            "name": "foo (bar, baz 01)",
            "path_to_pretrained_state_dict": None,
            "token_embedder": {
                "class": "BetaCdrEmbedder",
                "initargs": {"embedding_dim": D_MODEL},
            },
            "self_attention_stack": {
                "class": "SelfAttentionStackWithBuiltins",
                "initargs": {
                    "num_layers": NUM_LAYERS,
                    "d_model": D_MODEL,
                    "nhead": N_HEAD,
                },
            },
            "mlm_token_prediction_projector": {
                "class": "AminoAcidTokenProjector",
                "initargs": {"d_model": D_MODEL},
            },
            "vector_representation_delegate": {
                "class": "AveragePoolVectorRepresentationDelegate",
                "initargs": {},
            },
            "trainable_model": {"class": "MlmTrainableModel", "initargs": {}},
        },
        "data": {
            "path_to_training_data": "tests/resources/mock_data.csv",
            "path_to_validation_data": "tests/resources/mock_data.csv",
            "tokeniser": {"class": "BetaCdrTokeniser", "initargs": {}},
            "batch_collator": {"class": "MlmBatchCollator", "initargs": {}},
            "dataloader": {"initargs": {"batch_size": 3, "num_workers": 1}},
        },
        "loss": {
            "cross_entropy_loss": {
                "class": "AdjustedCELoss",
                "initargs": {"label_smoothing": 0.1},
            }
        },
        "optimiser": {"initargs": {"n_warmup_steps": 10, "d_model": D_MODEL}},
        "num_epochs": 3,
    }
