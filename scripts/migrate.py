"""scripts/migrate.py — Run DB migrations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()
from storage.postgres.audit import run_migrations
run_migrations()
print("✓ Migrations complete")
