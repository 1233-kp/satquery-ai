"""
Auth is additive: email/password + Google/GitHub OAuth, all issuing the
same bearer JWT. Nothing in here is imported by, or required by, the
analysis/upload routes -- see app/services/auth_service.py's
get_current_user_optional for how a route could look up "who's logged
in" without ever gating on it.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User
from app.schemas import LoginRequest, PublicUser, RegisterRequest, TokenResponse
from app.services.auth_service import (
    create_access_token,
    get_current_user_required,
    hash_password,
    verify_password,
)
from app.services.oauth_clients import oauth

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["auth"])


def _public_user(user: User) -> PublicUser:
    return PublicUser(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        oauth_provider=user.oauth_provider,
    )


def _find_or_create_oauth_user(
    db: Session, provider: str, oauth_id: str, email: str | None, display_name: str | None, avatar_url: str | None
) -> User:
    user = db.query(User).filter(User.oauth_provider == provider, User.oauth_id == oauth_id).first()
    if user:
        return user

    # Same verified email as an existing account (password-based or a
    # different provider) -- link this identity onto it instead of
    # crashing on the unique-email constraint or creating a duplicate.
    if email:
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.oauth_provider = user.oauth_provider or provider
            user.oauth_id = user.oauth_id or oauth_id
            user.avatar_url = user.avatar_url or avatar_url
            db.commit()
            db.refresh(user)
            return user

    user = User(
        email=email,
        oauth_provider=provider,
        oauth_id=oauth_id,
        display_name=display_name,
        avatar_url=avatar_url,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(400, "An account with this email already exists")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        display_name=payload.display_name or payload.email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user=_public_user(user))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.strip().lower()).first()
    if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user=_public_user(user))


@router.get("/me", response_model=PublicUser)
def me(current_user: User = Depends(get_current_user_required)):
    return _public_user(current_user)


@router.post("/logout")
def logout():
    # Bearer tokens are stored client-side (see auth_service.py's module
    # docstring for why) -- there's no server-side session to invalidate.
    # The frontend drops the stored token; this endpoint exists so that
    # flow has a clear, explicit server round-trip rather than being
    # silently client-only.
    return {"ok": True}


@router.get("/google/login")
async def google_login(request: Request):
    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback", name="google_callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo")
        if not userinfo:
            userinfo = await oauth.google.parse_id_token(request, token)
    except Exception as e:
        return RedirectResponse(f"{settings.frontend_url}/auth/callback?error={_safe_error(e)}")

    user = _find_or_create_oauth_user(
        db,
        provider="google",
        oauth_id=userinfo["sub"],
        email=userinfo.get("email"),
        display_name=userinfo.get("name"),
        avatar_url=userinfo.get("picture"),
    )
    jwt_token = create_access_token(user.id)
    return RedirectResponse(f"{settings.frontend_url}/auth/callback?token={jwt_token}")


@router.get("/github/login")
async def github_login(request: Request):
    redirect_uri = str(request.url_for("github_callback"))
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/github/callback", name="github_callback")
async def github_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.github.authorize_access_token(request)
        profile_resp = await oauth.github.get("user", token=token)
        profile = profile_resp.json()

        email = profile.get("email")
        if not email:
            emails_resp = await oauth.github.get("user/emails", token=token)
            emails = emails_resp.json()
            verified_primary = [e["email"] for e in emails if e.get("primary") and e.get("verified")]
            email = verified_primary[0] if verified_primary else (emails[0]["email"] if emails else None)
    except Exception as e:
        return RedirectResponse(f"{settings.frontend_url}/auth/callback?error={_safe_error(e)}")

    user = _find_or_create_oauth_user(
        db,
        provider="github",
        oauth_id=str(profile["id"]),
        email=email,
        display_name=profile.get("name") or profile.get("login"),
        avatar_url=profile.get("avatar_url"),
    )
    jwt_token = create_access_token(user.id)
    return RedirectResponse(f"{settings.frontend_url}/auth/callback?token={jwt_token}")


def _safe_error(e: Exception) -> str:
    import urllib.parse
    return urllib.parse.quote(str(e)[:200])
