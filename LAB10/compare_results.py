from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "runs" / "test_metrics.csv"
if not CSV.exists():
    raise FileNotFoundError("ไม่พบ runs/test_metrics.csv กรุณารัน evaluate.py ก่อน")

df = pd.read_csv(CSV)
print(df.to_string(index=False))

ax = df.set_index("experiment")[["precision", "recall", "mAP50", "mAP50_95"]].plot(
    kind="bar", figsize=(12, 6), ylim=(0, 1)
)
ax.set_ylabel("Score")
ax.set_title("YOLO Penguin Test Performance")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
out = ROOT / "runs" / "performance_comparison.png"
plt.savefig(out, dpi=200)
print("บันทึกกราฟ:", out)
