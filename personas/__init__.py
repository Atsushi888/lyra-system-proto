# personas/__init__.py

# パッケージ読み込み時に Floria の persona モジュールを読み込むだけにする
from . import persona_floria_ja

__all__ = [
    "persona_floria_ja",
]
