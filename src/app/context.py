import logging

import requests
from requests.adapters import HTTPAdapter
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from urllib3.util.retry import Retry

from .progress import RichProgressSink

rich_progress = Progress(
    SpinnerColumn(),
    TextColumn("{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
)

progress = RichProgressSink(rich_progress)

logger = logging.getLogger("pharm2d")
logger.setLevel(logging.DEBUG)
logger.propagate = False

file_handler = logging.FileHandler("pharm2d.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

session = requests.Session()

retry = Retry(
    total=5,
    connect=5,
    read=5,
    status=5,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("HEAD", "GET", "OPTIONS"),
    raise_on_status=False,
)

adapter = HTTPAdapter(
    max_retries=retry,
    pool_connections=20,
    pool_maxsize=20,
)

session.mount("http://", adapter)
session.mount("https://", adapter)

session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
)
