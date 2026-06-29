"""
Shared FastAPI dependencies used by every protected router.

get_current_user   — extracts and validates the JWT Bearer token,
                     returns the authenticated User ORM object.
require_roles      — factory that returns a dependency raising HTTP 403
                     if the authenticated user's role is not in the
                     allowed set.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.security import decode_token
from app.db.models.user import User, UserRole
from app.db.session import get_db

bearer = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        if not user_id:
            raise ValueError("missing sub")
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    user = db.get(User, uuid.UUID(user_id))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )
    return user


def require_roles(*roles: UserRole):
    """
    Returns a FastAPI dependency that raises HTTP 403 if the current
    user's role is not in the provided set.

    Usage:
        @router.get("/admin-only")
        def admin_endpoint(user = Depends(require_roles(UserRole.ADMIN))):
            ...
    """
    def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access requires one of: {[r.value for r in roles]}",
            )
        return user
    return _check
