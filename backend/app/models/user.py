"""
User models for authentication and authorization.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    PRACTITIONER = "practitioner"
    RESEARCHER = "researcher"


class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole = UserRole.PRACTITIONER
    specialty: Optional[str] = None  # e.g., "Oncologist", "General Practitioner"
    institution: Optional[str] = None


class UserCreate(UserBase):
    """Model for creating a new user."""
    password: str = Field(..., min_length=8)


class UserInDB(UserBase):
    """User model as stored in database."""
    id: Optional[str] = Field(None, alias="_id")
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True


class UserResponse(UserBase):
    """User model for API responses (no password)."""
    id: str
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        populate_by_name = True


class User(UserBase):
    """User model for internal use."""
    id: str
    is_active: bool = True


class Token(BaseModel):
    """JWT Token model."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None

