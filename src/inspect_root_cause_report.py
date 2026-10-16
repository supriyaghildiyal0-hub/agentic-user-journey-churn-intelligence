
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

report_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "visitor_root_cause_report.csv"
)

df = pd.read_csv(report_path, nrows=5)

print("\n========== REPORT CHECK ==========")
print("File found:", report_path.exists())
print("Columns available:")
for column in df.columns:
    print("-", column)

print("\n========== SAMPLE VISITOR RECORD ==========")
print(df.head(1).T.to_string())

print("\n========== CHECK COMPLETE ==========")
