"""Allow running as `python -m das_audit`."""
from das_audit.cli import main
import sys

sys.exit(main())
