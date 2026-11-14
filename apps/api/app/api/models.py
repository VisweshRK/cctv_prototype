"""Model Registry and Commercial Provenance Endpoints.
"""

from fastapi import APIRouter
from apps.api.app.core.runtime import runtime

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("")
def list_models():
    """Returns commercial licensing audit, model provenance, and validation status."""
    return runtime.model_registry.list_all()
