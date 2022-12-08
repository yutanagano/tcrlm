#!/bin/bash

python3 -m pytest -v tests/0_unit_tests --cov=src
python3 -m pytest -v tests/1_training_loop_tests/test_mlm.py
python3 -m pytest -v tests/1_training_loop_tests/test_us_simcl.py