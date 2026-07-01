import sys
sys.path.insert(0, "src")
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from eurocoin_research.data.panel import PanelAssembler
from eurocoin_research.config import load_config

config = load_config()
assembler = PanelAssembler(config=config)
raw = assembler.fetch_all()
panel = assembler.assemble_panel(raw_data=raw)
assembler.save_panel(panel)

print(f"\n=== Panel Summary ===")
print(f"Months: {len(panel)}")
print(f"Series: {len(panel.columns)-1}")
print(f"Date range: {panel['date'].min()} -> {panel['date'].max()}")
print(f"\nNon-null counts:")
for col in panel.columns:
    if col != "date":
        n = panel[col].is_not_null().sum()
        print(f"  {col:15s}: {n:4d}")
