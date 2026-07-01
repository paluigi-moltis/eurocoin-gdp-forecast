import sys
sys.path.insert(0, "src")
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from eurocoin_research.config import load_config
from eurocoin_research.data.loaders.ecb import ECBLoader

config = load_config()
loader = ECBLoader(cache_dir=None)

for spec in config.sources["ecb"].series:
    print(f"\n=== {spec.id}: dataflow={spec.code}, key={spec.filter} ===")
    try:
        df = loader.fetch_series(spec, start="2020-01")
        print(f"  {len(df)} observations [{df['date'].min()} -> {df['date'].max()}]")
        if len(df) > 0:
            print(f"  Latest: {df.tail(3)}")
    except Exception as e:
        print(f"  ERROR: {e}")
