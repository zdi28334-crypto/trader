"""支持 `python -m trader ...`。这种写法不依赖可编辑安装的 .pth 文件，在 macOS 上更稳。"""

from trader.cli import main

main()
