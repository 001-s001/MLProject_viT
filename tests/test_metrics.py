from pathlib import Path

from src.utils.metrics import MetricsLogger


def test_metrics_logger_writes_expected_csv_columns(tmp_path: Path):
    logger = MetricsLogger(tmp_path / "metrics.csv")

    logger.log(
        epoch=1,
        train_loss=1.25,
        val_loss=1.5,
        val_acc=42.0,
        best_acc=42.0,
        lr=0.0003,
    )

    text = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "epoch,train_loss,val_loss,val_acc,best_acc,lr"
    assert text.splitlines()[1] == "1,1.25,1.5,42.0,42.0,0.0003"

