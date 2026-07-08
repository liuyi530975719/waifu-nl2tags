"""waifu-nl2tags — bilingual natural-language -> Illustrious/NoobAI tag translator."""
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("waifu-nl2tags")
except Exception:
    __version__ = "0.6.2"
from .illustrious import default_formatter, Formatter          # noqa: F401
from .infer import translate, load_model                        # noqa: F401
