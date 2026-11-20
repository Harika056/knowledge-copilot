from __future__ import annotations
import time

import jwt

from .config import ROLE_ACCESS


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def create_token(sub: str, role: str, secret: str, ttl_s: int = 900) -> str:
    return jwt.encode({"sub": sub, "role": role, "exp": int(time.time()) + ttl_s}, secret, algorithm="HS256")


def verify_token(token: str, secret: str) -> dict:
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token expired") from e
    except jwt.PyJWTError as e:
        raise AuthError("invalid token") from e
    if claims.get("role") not in ROLE_ACCESS:
        raise AuthError("unknown role", status=403)
    return claims
