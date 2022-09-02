from copy import deepcopy
from tests.resources.mockups import MockDistributedDataParallel, MockDevice
import pytest
from source.nn import models
from source.utils import fileio
import torch


@pytest.fixture
def dummy_pretrain_model():
    bert = models.Cdr3Bert(
        num_encoder_layers=2,
        d_model=2,
        nhead=2,
        dim_feedforward=4
    )
    model = models.Cdr3BertPretrainWrapper(
        bert=bert
    )

    return model


@pytest.fixture
def dummy_finetune_model():
    alpha_bert = models.Cdr3Bert(
        num_encoder_layers=2,
        d_model=2,
        nhead=2,
        dim_feedforward=4
    )
    beta_bert = deepcopy(alpha_bert)
    embedder = models.TcrEmbedder(alpha_bert=alpha_bert, beta_bert=beta_bert)
    model = models.Cdr3BertFineTuneWrapper(tcr_embedder=embedder)

    return model


def state_dicts_equivalent(
    state_dict_1: dict,
    state_dict_2: dict
) -> bool:
    if len(state_dict_1) != len(state_dict_2):
        return False
    
    for key in state_dict_1:
        if not state_dict_1[key].equal(state_dict_2[key]):
            return False

    return True


class TestResolvedPathFromMaybeStr:
    def test_path(self, tmp_path):
        assert fileio.resolved_path_from_maybe_str(tmp_path) == \
            tmp_path.resolve()
    

    def test_str(self, tmp_path):
        assert fileio.resolved_path_from_maybe_str(str(tmp_path)) == \
            tmp_path.resolve()


    def test_bad_type(self):
        with pytest.raises(RuntimeError):
            fileio.resolved_path_from_maybe_str(1)


class TestCreateTrainingRunDirectory:
    def test_new_pretrain_directory(self, tmp_path):
        result = fileio.create_training_run_directory(
            working_directory=tmp_path,
            run_id='foo',
            mode='pretrain'
        )
        expected = tmp_path / 'pretrain_runs' / 'foo'

        assert result == expected
        assert expected.is_dir()


    def test_new_finetune_directory(self, tmp_path):
        result = fileio.create_training_run_directory(
            working_directory=tmp_path,
            run_id='foo',
            mode='finetune'
        )
        expected = tmp_path / 'finetune_runs' / 'foo'

        assert result == expected
        assert expected.is_dir()


    def test_overwrite_clash(self, tmp_path):
        (tmp_path / 'pretrain_runs').mkdir()
        (tmp_path / 'pretrain_runs' / 'foo').mkdir()
        (tmp_path / 'pretrain_runs' / 'foo' / 'bar').mkdir()
        
        result = fileio.create_training_run_directory(
            working_directory=tmp_path,
            run_id='foo',
            mode='pretrain',
            overwrite=True
        )
        expected = tmp_path / 'pretrain_runs' / 'foo'

        assert result == expected
        assert expected.is_dir()
        assert not (expected / 'bar').is_dir()
    

    def test_overwrite_adjust(self, tmp_path):
        (tmp_path / 'pretrain_runs').mkdir()
        (tmp_path / 'pretrain_runs' / 'foo').mkdir()
        (tmp_path / 'pretrain_runs' / 'foo_1').mkdir()

        result = fileio.create_training_run_directory(
            working_directory=tmp_path,
            run_id='foo',
            mode='pretrain',
            overwrite=False
        )
        expected = tmp_path / 'pretrain_runs' / 'foo_2'

        assert result == expected
        assert expected.is_dir()


    def test_nonexistent_working_directory(self):
        with pytest.raises(RuntimeError):
            fileio.create_training_run_directory(
                working_directory='foobarbaz',
                run_id='foo',
                mode='p'
            )


    def test_bad_mode(self, tmp_path):
        with pytest.raises(RuntimeError):
            fileio.create_training_run_directory(
                working_directory=tmp_path,
                run_id='foo',
                mode='bar'
            )


class TestTrainingRecordManager:
    def test_init_nonexistent_training_run_directory(self):
        with pytest.raises(RuntimeError):
            fileio.TrainingRecordManager(
                training_run_dir='foobarbaz',
                distributed=False,
                device=MockDevice('cpu')
            )


    @pytest.mark.parametrize(
        ('distributed', 'device', 'expected_logname'),
        (
            (False, MockDevice('cpu'), 'training_log.csv'),
            (True, MockDevice('cuda:0'), 'training_log_cuda_0.csv')
        )
    )
    def test_save_log(
        self,
        tmp_path,
        distributed,
        device,
        expected_logname
    ):
        manager = fileio.TrainingRecordManager(
            training_run_dir=tmp_path,
            distributed=distributed,
            device=device
        )
        log = {
            0: {'foo': 'foo', 'bar': 'bar'},
            1: {'foo': 'foo', 'bar': 'bar'},
            2: {'foo': 'foo', 'bar': 'bar'}
        }

        manager.save_log(log_dict=log)

        expected_path = tmp_path / expected_logname
        expected_contents = 'epoch,foo,bar\n0,foo,bar\n1,foo,bar\n2,foo,bar\n'

        assert expected_path.is_file()

        with open(expected_path, 'r') as f:
            result_contents = f.read()
        
        assert result_contents == expected_contents
    

    def test_save_model_pretrained(self, tmp_path, dummy_pretrain_model):
        manager = fileio.TrainingRecordManager(
            training_run_dir=tmp_path,
            distributed=False,
            device=MockDevice('cpu'),
            test_mode=False
        )

        manager.save_model(model=dummy_pretrain_model)

        bert_state_dict_path = tmp_path/'bert_state_dict.pt'
        generator_state_dict_path = tmp_path/'generator_state_dict.pt'

        assert bert_state_dict_path.is_file()
        assert generator_state_dict_path.is_file()

        bert_state_dict = torch.load(bert_state_dict_path)
        generator_state_dict = torch.load(generator_state_dict_path)

        assert state_dicts_equivalent(
            dummy_pretrain_model.bert.state_dict(),
            bert_state_dict
        )
        assert state_dicts_equivalent(
            dummy_pretrain_model.generator.state_dict(),
            generator_state_dict
        )


    def test_save_model_finetuned(self, tmp_path, dummy_finetune_model):
        manager = fileio.TrainingRecordManager(
            training_run_dir=tmp_path,
            distributed=False,
            device=MockDevice('cpu'),
            test_mode=False
        )

        manager.save_model(model=dummy_finetune_model)

        alpha_bert_state_dict_path = tmp_path/'alpha_bert_state_dict.pt'
        beta_bert_state_dict_path = tmp_path/'beta_bert_state_dict.pt'
        classifier_state_dict_path = tmp_path/'classifier_state_dict.pt'

        assert alpha_bert_state_dict_path.is_file()
        assert beta_bert_state_dict_path.is_file()
        assert classifier_state_dict_path.is_file()

        alpha_bert_state_dict = torch.load(alpha_bert_state_dict_path)
        beta_bert_state_dict = torch.load(beta_bert_state_dict_path)
        classifier_state_dict = torch.load(classifier_state_dict_path)

        assert state_dicts_equivalent(
            dummy_finetune_model.embedder.alpha_bert.state_dict(),
            alpha_bert_state_dict
        )
        assert state_dicts_equivalent(
            dummy_finetune_model.embedder.beta_bert.state_dict(),
            beta_bert_state_dict
        )
        assert state_dicts_equivalent(
            dummy_finetune_model.classifier.state_dict(),
            classifier_state_dict
        )


    @pytest.mark.parametrize(
        'device', (MockDevice('cuda:0'), MockDevice('cuda:1'))
    )
    def test_save_model_distributed(
        self,
        tmp_path,
        dummy_pretrain_model,
        device
    ):
        manager = fileio.TrainingRecordManager(
            training_run_dir=tmp_path,
            distributed=True,
            device=device,
            test_mode=False
        )

        manager.save_model(
            model=MockDistributedDataParallel(module=dummy_pretrain_model)
        )

        if device.index == 1:
            assert len(list(tmp_path.iterdir())) == 0
            return

        bert_state_dict_path = tmp_path/'bert_state_dict.pt'
        generator_state_dict_path = tmp_path/'generator_state_dict.pt'

        assert bert_state_dict_path.is_file()
        assert generator_state_dict_path.is_file()

    
    def test_save_model_test_mode(self, tmp_path, dummy_pretrain_model):
        manager = fileio.TrainingRecordManager(
            training_run_dir=tmp_path,
            distributed=False,
            device=MockDevice('cuda:0'),
            test_mode=True
        )

        manager.save_model(model=dummy_pretrain_model)

        bert_state_dict_path = tmp_path/'bert_state_dict_cuda_0.pt'
        generator_state_dict_path = tmp_path/'generator_state_dict_cuda_0.pt'

        assert bert_state_dict_path.is_file()
        assert generator_state_dict_path.is_file()


class TestBoolConvert:
    def test_true(self):
        assert fileio._bool_convert(x='True')


    def test_false(self):
        assert not fileio._bool_convert(x='False')


    def test_bad_value(self):
        with pytest.raises(RuntimeError):
            fileio._bool_convert('T')


class TestParseHyperparams:
    def test_parsing(self):
        expected = {
            'path_train_data': 'tests/resources/data/mock_unlabelled.csv',
            'path_valid_data': 'tests/resources/data/mock_unlabelled.csv',
            'num_encoder_layers': 16,
            'd_model': 16,
            'nhead': 4,
            'dim_feedforward': 128,
            'activation': 'gelu',
            'train_batch_size': 6,
            'valid_batch_size': 6,
            'batch_optimisation': True,
            'lr': 0.001,
            'lr_decay': True,
            'optim_warmup': 5,
            'num_epochs': 3,
            'foo': False
        }
        result = fileio.parse_hyperparams(
            'tests/resources/hyperparams/pretrain.csv'
        )

        assert result == expected
    

    def test_nonexistent_file(self):
        with pytest.raises(RuntimeError):
            fileio.parse_hyperparams('foo/bar/baz.csv')


    def test_noncsv_file(self):
        with pytest.raises(RuntimeError):
            fileio.parse_hyperparams('tests/0_unit_tests/test_utils_fileio.py')


    def test_bad_format(self):
        with pytest.raises(RuntimeError):
            fileio.parse_hyperparams('tests/resources/data/bad_format.csv')


    def test_bad_types(self):
        with pytest.raises(RuntimeError):
            fileio.parse_hyperparams(
                'tests/resources/hyperparams/bad_types.csv'
            )


    def test_bad_values(self):
        with pytest.raises(RuntimeError):
            fileio.parse_hyperparams(
                'tests/resources/hyperparams/bad_values.csv'
            )