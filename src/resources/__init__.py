import json
from pkg_resources import resource_stream

with resource_stream(__name__, 'v_cdrs.json') as f:
    V_CDRS = json.load(f)