"""adapters 层：模型调用适配器。"""

from app.adapters.base import (
    BaseAdapter,
    ModelAdapter,
    RawModelOutput,
    get_adapter,
    register_adapter,
)

__all__ = ["BaseAdapter", "ModelAdapter", "RawModelOutput", "get_adapter", "register_adapter"]
