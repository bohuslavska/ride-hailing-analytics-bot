"""
HTTP layer.

Deliberately does not re-export `app` from src.api.app. Binding that name here
shadows the submodule of the same name, so `src.api.app` then resolves to the
FastAPI instance rather than the module, and anything addressing the module by
path -- patching in tests, for one -- silently gets the wrong object. Entry points
use the `src.api.app:app` form instead.
"""
