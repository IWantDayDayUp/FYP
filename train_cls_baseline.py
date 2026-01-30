# train_cls_baseline.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_fm.training.cls_baseline import build_cls_argparser, train_classifier


def main() -> None:
    parser = build_cls_argparser()
    args = parser.parse_args()

    train_classifier(args)


if __name__ == "__main__":
    main()
