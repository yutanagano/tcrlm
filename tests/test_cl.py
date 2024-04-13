from libtcrlm.token_embedder import SingleChainCdrEmbedder
from libtcrlm.self_attention_stack import SelfAttentionStackWithBuiltins
from libtcrlm.mlm_token_prediction_projector import AminoAcidTokenProjector
from libtcrlm.vector_representation_delegate import AveragePoolVectorRepresentationDelegate
from libtcrlm.bert import Bert
from multiprocessing import Process
import pytest
from pytest import mark
from src.model_trainer import ModelTrainer
import torch
import tests.resources.helper_functions as helper_functions
import warnings


D_MODEL = 4
NUM_LAYERS = 2
N_HEAD = 2


@mark.slow
@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
def test_main(model_template, config, tmp_path):
    if not torch.cuda.is_available():
        warnings.warn("CL test skipped due to hardware limitations")
        return

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
            "valid_cont_loss",
            "valid_positive_distance",
            "valid_negative_distance",
            "valid_mlm_loss",
            "valid_mlm_acc",
        ],
        expected_len=4,
    )
    assert helper_functions.config_saved(expected_model_save_dir, config)


@pytest.fixture
def model_template():
    token_embedder = SingleChainCdrEmbedder(embedding_dim=D_MODEL)
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
        "training_delegate": {"class": "ClTrainingDelegate", "initargs": {}},
        "model": {
            "name": "foo (bar, baz 01)",
            "path_to_pretrained_state_dict": None,
            "token_embedder": {
                "class": "SingleChainCdrEmbedder",
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
            "trainable_model": {"class": "ClTrainableModel", "initargs": {}},
        },
        "data": {
            "training_data": {
                "dataset": {
                    "class": "TcrDataset",
                    "initargs": {}
                },
                "dataloader": {
                    "class": "SingleDatasetDataLoader",
                    "initargs": {"batch_size": 3, "num_workers": 1},
                },
                "csv_paths": ["tests/resources/mock_data.csv"],
            },
            "validation_data": {
                "dataset": {
                    "class": "TcrDataset",
                    "initargs": {}
                },
                "dataloader": {
                    "class": "SingleDatasetDataLoader",
                    "initargs": {"batch_size": 3, "num_workers": 1},
                },
                "csv_paths": ["tests/resources/mock_data.csv"],
            },
            "tokeniser": {"class": "BetaCdrTokeniser", "initargs": {}},
            "batch_collator": {"class": "ClBatchCollator", "initargs": {"prob_drop_chain": 0}},
        },
        "loss": {
            "cross_entropy_loss": {
                "class": "AdjustedCrossEntropyLoss",
                "initargs": {"label_smoothing": 0.1},
            },
            "contrastive_loss": {
                "class": "EuclideanDistanceLoss",
                "initargs": {"temp": 0.05},
            },
        },
        "optimiser": {"initargs": {"n_warmup_steps": 10, "d_model": D_MODEL}},
        "num_epochs": 3,
    }
