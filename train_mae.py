# train_mae.py
from pathlib import Path
import sys

# TEMP: ensure "src" is importable when running from repo root
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_fm.training.mae_single import build_mae_argparser, train_mae_single


def main() -> None:
    parser = build_mae_argparser()
    args = parser.parse_args()
    train_mae_single(args)


if __name__ == "__main__":
    main()
