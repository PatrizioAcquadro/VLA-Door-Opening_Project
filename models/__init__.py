"""
Model architectures module.

This module contains:
- Transformer architectures
- World model components
- Flow matching modules
- VLM backbone (lazy import; requires ``pip install -e ".[vlm]"``)
- VLA model (lazy import; requires ``pip install -e ".[vlm]"``)

The VLM backbone and VLA model are not included in __all__ because they require
the optional ``transformers`` dependency. Import them directly when needed:
    from models.vlm_backbone import VLMBackbone, load_vlm_backbone
    from models.vla_model import VLAModel, load_vla_model
"""

from models.transformer import TransformerModel
from models.utils import count_parameters, format_params, get_model

__all__ = ["TransformerModel", "count_parameters", "format_params", "get_model"]
