"""Central configuration: paths, simulation parameters, and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
POWERBI = ROOT / "powerbi"

for _p in (RAW, PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261024

# Experiment design
N_USERS = 60_000
PRE_DAYS = 14            # days -14..-1: pre-period (used for CUPED + balance checks)
EXP_DAYS = 28            # days 0..27: experiment
START_DATE = "2026-08-03"  # day 0 (a Monday)
ARMS = ["control", "ai_feed", "ai_feed_social"]
ARM_LABELS = {
    "control": "Control (ranked-chronological)",
    "ai_feed": "AI 'For You' feed",
    "ai_feed_social": "AI feed + friend boost",
}
TREATMENTS = ["ai_feed", "ai_feed_social"]
ALPHA = 0.05
CHURN_WINDOW = 7         # churned = no activity in the final 7 experiment days

# Business assumptions (all configurable, used by the impact model)
MAU_POPULATION = 40_000_000   # users eligible for rollout
ANNUAL_DAYS = 365
LTV_PER_RETAINED_USER = 38.0  # USD, 12-month net ad revenue of an average retained user
RETENTION_CAMPAIGN_COST = 0.60   # USD per targeted user (push + incentive)
RETENTION_CAMPAIGN_SAVE_RATE = 0.18  # share of true churners saved when targeted

# Palette (validated categorical slots 1-3, see reports/README_palette note)
ARM_COLORS = {"control": "#2a78d6", "ai_feed": "#eb6834", "ai_feed_social": "#1baf7a"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MONTHLY_BASE_CHURN = 0.035    # platform-wide monthly churn of mature users (measured HRs scale it)
MC_DRAWS = 20_000

# Power BI reads its CSVs from here (raw.githubusercontent.com/<this path><table>.csv)
GITHUB_DATA_PATH = "NirmalKumar31/social-engagement-intelligence-pro/main/powerbi/data/"
