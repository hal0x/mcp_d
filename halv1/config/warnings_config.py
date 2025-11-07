"""Конфигурация для подавления предупреждений SWIG от faiss-cpu."""

import warnings

# Подавляем предупреждения SWIG от faiss-cpu
warnings.filterwarnings("ignore", message=".*SwigPyPacked has no __module__ attribute.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SwigPyObject has no __module__ attribute.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*swigvarlink has no __module__ attribute.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type SwigPyPacked has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type SwigPyObject has no __module__ attribute", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="builtin type swigvarlink has no __module__ attribute", category=DeprecationWarning)
