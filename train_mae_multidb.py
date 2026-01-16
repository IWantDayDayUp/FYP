# train_mae_multidb.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_fm.training.mae_multidb import build_mae_multidb_argparser, train_mae_multidb


def main() -> None:
    parser = build_mae_multidb_argparser()
    args = parser.parse_args()
    train_mae_multidb(args)


if __name__ == "__main__":
    main()
