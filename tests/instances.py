from pathlib import Path

from erp.instance import load_instance

ROOT = Path(__file__).resolve().parents[1]
A = load_instance(ROOT / "data" / "instance_2x2.json")
B = load_instance(ROOT / "data" / "instance_3x3.json")
DUE = load_instance(ROOT / "data" / "instance_3x3_due.json")
