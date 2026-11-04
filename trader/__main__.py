"""支持 `python -m trader 600519 --start 2020-01-01`。

这种写法不依赖可编辑安装的 .pth 文件，在 macOS 上更稳。
"""

from trader.cli import fetch_main

fetch_main()
