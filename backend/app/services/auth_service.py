"""
Authentication service for user management and JWT handling.
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, User as UserDB
from app.models.user import UserCreate, UserInDB, User, Token, TokenData, UserRole

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme (kept for OpenAPI schema, but we manually parse the header)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_v1_prefix}/auth/login")


class AuthService:
    """Service for authentication and authorization."""

    @staticmethod
    def _truncate_password(password: str) -> str:
        """Truncate password to 72 bytes (bcrypt limit)."""
        # bcrypt has a 72 byte limit for passwords
        return password.encode('utf-8')[:72].decode('utf-8', errors='ignore')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        truncated = AuthService._truncate_password(plain_password)
        return pwd_context.verify(truncated, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password."""
        truncated = AuthService._truncate_password(password)
        return pwd_context.hash(truncated)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[UserDB]:
        """Get a user by email from database."""
        return db.query(UserDB).filter(UserDB.email == email).first()

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[UserDB]:
        """Get a user by ID from database."""
        return db.query(UserDB).filter(UserDB.id == user_id).first()

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[UserDB]:
        """Authenticate a user with email and password."""
        user = AuthService.get_user_by_email(db, email)
        if not user or not AuthService.verify_password(password, user.hashed_password):
            return None
        return user

    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> UserDB:
        """Create a new user in database."""
        # Check if user exists
        existing = AuthService.get_user_by_email(db, user_data.email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user - convert enum to string value
        role_value = user_data.role.value if hasattr(user_data.role, 'value') else str(user_data.role)

        db_user = UserDB(
            email=user_data.email,
            hashed_password=AuthService.get_password_hash(user_data.password),
            full_name=user_data.full_name,
            role=role_value,
            specialty=user_data.specialty,
            institution=user_data.institution,
            is_active=True
        )

        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        return db_user


def get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> User:
    """Dependency to get current authenticated user from JWT token.

    We manually parse the ``Authorization: Bearer <token>`` header instead of relying
    on ``OAuth2PasswordBearer`` because, in some environments, it was returning
    ``"Not authenticated"`` even when the header was present. This implementation
    is explicit and more robust for our frontend and CLI clients.
    """
    import logging
    logger = logging.getLogger(__name__)

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization:
        logger.error("Missing Authorization header")
        raise credentials_exception

    # Expect header of the form: "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.error(f"Invalid Authorization header format: {authorization}")
        raise credentials_exception

    token = parts[1]

    try:
        logger.info("Decoding JWT token from Authorization header...")
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: str = payload.get("sub")
        logger.info(f"Token decoded. User ID: {user_id}")
        if user_id is None:
            logger.error("No 'sub' (user_id) claim in token")
            raise credentials_exception
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise credentials_exception

    user = AuthService.get_user_by_id(db, user_id)
    if user is None:
        logger.error(f"User not found: {user_id}")
        raise credentials_exception

    logger.info(f"User authenticated: {user.email}")
    return User(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        specialty=user.specialty,
        institution=user.institution,
        is_active=user.is_active
    )


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to ensure user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

