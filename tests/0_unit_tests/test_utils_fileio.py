from tests.resources.mockups import MockDistributedDataParallel
import pytest
import source.utils.fileio as fileio
import torch


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


class TestWriteHyperparams:
    def test_write_hyperparams(self, tmp_path):
        hyperparams = {
            'foo': 'bar',
            'baz': 'bat'
        }

        fileio.write_hyperparameters(
            hyperparameters=hyperparams,
            training_run_dir=tmp_path
        )

        assert (tmp_path / 'hyperparams.txt').is_file()

        with open(tmp_path / 'hyperparams.txt', 'r') as f:
            for param in hyperparams:
                line = f.readline()
                assert line == f'{param}: {hyperparams[param]}\n'


    def test_nonexistent_training_run_dir(self):
        with pytest.raises(RuntimeError):
            fileio.write_hyperparameters(
                hyperparameters={'foo': 'bar'},
                training_run_dir='foobarbaz'
            )


class TestTrainingRecordManager:
    def test_init_nonexistent_training_run_directory(self):
        with pytest.raises(RuntimeError):
            fileio.TrainingRecordManager(
                training_run_dir='foobarbaz',
                distributed=False,
                device=torch.device('cpu')
            )


    @pytest.mark.parametrize(
        ('distributed', 'device', 'expected_logname'),
        (
            (False, torch.device('cpu'), 'training_log.csv'),
            (True, torch.device('cuda:0'), 'training_log_cuda_0.csv')
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
    

    @pytest.mark.parametrize(
        ('distributed', 'test_mode', 'device', 'expected_filename'),
        (
            (False, False, torch.device('cpu'), 'model.ptnn'),
            (False, True, torch.device('cpu'), 'model.ptnn'),
            (True, False, torch.device('cuda:0'), 'model.ptnn'),
            (True, False, torch.device('cuda:1'), None),
            (True, True, torch.device('cuda:0'), 'model_cuda_0.ptnn')
        )
    )
    def test_save_model(
        self,
        tmp_path,
        distributed,
        test_mode,
        device,
        expected_filename
    ):
        manager = fileio.TrainingRecordManager(
            training_run_dir=tmp_path,
            distributed=distributed,
            device=device,
            test_mode=test_mode
        )

        model = torch.nn.Linear(3,3)
        if distributed:
            model = MockDistributedDataParallel(model)

        def equivalent(model1: torch.nn.Module, model2: torch.nn.Module):
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                if not torch.equal(p1, p2):
                    return False
            return True
        
        manager.save_model(model=model, name='model')

        if expected_filename is None:
            assert len(list(tmp_path.iterdir())) == 0
            return

        expected_path = tmp_path / expected_filename
        assert expected_path.is_file()
        result_model = torch.load(expected_path)      
        assert equivalent(model, result_model)


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