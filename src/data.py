'''
data.py -- Module 1: data acquisition, cleaning, and per-expiry forward implication.

Pipeline position:
    fetch_chain / load_snapshot
    attach_forwards (implied_forward per expiry, OTM filter)
    [Module 2: iv_solver]
'''


from __future__ import annotations

import io
import logging
import time
from typing import Optional 

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)