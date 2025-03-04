import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

NAN_VALUE = -1000

PROJECT_PATH = Path(__file__).parent
MODEL_ROOT = PROJECT_PATH / "models"
DATA_ROOT = PROJECT_PATH / "data"
# print(MODEL_ROOT)
# print(DATA_ROOT)
# print(PROJECT_PATH)
# print(os.getcwd())
LOGS_ROOT = PROJECT_PATH / "logs"
CONFIGS_ROOT = PROJECT_PATH / "configs"
RESULT_PATH = PROJECT_PATH / "results"
sample_size = 100000 * 3

# MODEL_ROOT = Path(os.environ["MODEL_ROOT"])
# DATA_ROOT = Path(os.environ["DATA_ROOT"])

COMPUTE_NAIVE_CARDINALITY = False
