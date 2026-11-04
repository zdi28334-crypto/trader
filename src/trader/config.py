"""项目配置：路径等。可以用环境变量覆盖。"""

import os
from pathlib import Path

# 数据缓存目录，默认在项目根目录下的 data_cache/
CACHE_DIR = Path(os.environ.get("TRADER_CACHE_DIR", "data_cache"))
