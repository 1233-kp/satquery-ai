"""
Password hashing, JWT issuance/verification, and the current-user
dependencies. Kept deliberately separate from the OAuth client
registration (app/services/oauth_clients.py) and the auth router itself
so each piece is independently testable.

Design choice: sessions are bearer JWTs handed back in the OAuth
redirect's query string (?token=...) and stored client-side, not
httpOnly cookies. The OAuth handshake itself (state/nonce) still uses a
short-lived Starlette session cookie, but that cookie never leaves the
backend's own origin (both /login and /callback are backend routes), so
it never needs SameSite=None or cross-site cookie config at all. This
sidesteps cookie/CORS fragility entirely for the app's actual auth
session, which matters given the explicit design goal that auth must
never be the thing that breaks on demo day.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User

settings = get_settings()
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _pwd_context.verify(password, hashed)


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_user_id(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None
    return payload.get("sub")


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Returns the authenticated User, or None -- never raises. This is
    the dependency every analysis/upload route should use if it ever
    wants to know who's logged in, precisely so auth can never become a
    hard gate on the core demo path."""
    if credentials is None:
        return None
    user_id = decode_user_id(credentials.credentials)
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def get_current_user_required(
    user: Optional[User] = Depends(get_current_user_optional),
) -> User:
    """For the handful of endpoints (like /auth/me) that genuinely need a
    logged-in user. Never used on analysis/upload routes."""
    from fastapi import HTTPException

    if user is None:
        raise HTTPException(401, "Not authenticated")
    return user
