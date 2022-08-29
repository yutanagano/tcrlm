#!/bin/bash

python3 -m pytest -v tests/0_unit_tests --cov=source
python3 -m pytest -v tests/1_integration_tests