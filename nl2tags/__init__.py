"""waifu-nl2tags — bilingual natural-language -> Illustrious/NoobAI tag translator."""
__version__ = "0.1.0"
from .illustrious import default_formatter, Formatter          # noqa: F401
from .infer import translate, load_model                        # noqa: F401
