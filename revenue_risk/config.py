"""Centralized configuration, paths and safety limits."""
from __future__ import annotations
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

def load_dotenv() -> None:
    """Minimal .env loader, never overwrites a real environment variable."""
    path = ROOT / '.env'
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")

load_dotenv()
DATA_ZIP = Path(os.getenv('OLIST_ZIP', str(ROOT / 'data' / 'brazil.zip')))
DB_PATH = Path(os.getenv('RISK_DB', str(ROOT / 'runtime' / 'revenue.sqlite')))
RESULTS = Path(os.getenv('RISK_RESULTS', str(ROOT / 'results')))
ASPECTS = (
    'delivery_delay', 'product_not_received', 'damaged_item',
    'wrong_or_missing_item', 'product_quality', 'not_as_described',
    'seller_service', 'refund_or_return', 'packaging', 'price_value', 'other'
)
MAX_TOOL_CALLS = 4
MAX_EXAMPLES = 10
DEFAULT_SAMPLE_SIZE = 4000
