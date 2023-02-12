import json
from pkg_resources import resource_stream

AMINO_ACIDS = (
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y'
)

with resource_stream(__name__, 'functional_trajs.json') as f:
    FUNCTIONAL_TRAJS = tuple(json.load(f))
with resource_stream(__name__, 'functional_travs.json') as f:
    FUNCTIONAL_TRAVS = tuple(json.load(f))
with resource_stream(__name__, 'functional_trbjs.json') as f:
    FUNCTIONAL_TRBJS = tuple(json.load(f))
with resource_stream(__name__, 'functional_trbvs.json') as f:
    FUNCTIONAL_TRBVS = tuple(json.load(f))

with resource_stream(__name__, 'v_cdrs.json') as f:
    V_CDRS = json.load(f)