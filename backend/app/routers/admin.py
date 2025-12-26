"""
Admin routes for user management and audit logs.
"""
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db, User as UserDB, AuditLog as AuditLogDB
from app.models.user import User, UserResponse, UserRole
from app.services.auth_service import get_current_active_user, AuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Dependency to ensure user is admin."""
    if current_user.role != "admin" and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.get("/users", response_model=List[UserResponse])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: str = Query(None),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """List all users (admin only)."""
    query = db.query(UserDB)
    
    if search:
        query = query.filter(
            (UserDB.email.ilike(f"%{search}%")) |
            (UserDB.full_name.ilike(f"%{search}%"))
        )
    
    users = query.order_by(desc(UserDB.created_at)).offset(skip).limit(limit).all()
    
    return [
        UserResponse(
            id=str(u.id),
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            specialty=u.specialty,
            institution=u.institution,
            is_active=u.is_active,
            created_at=u.created_at
        )
        for u in users
    ]


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """Get a specific user (admin only)."""
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        specialty=user.specialty,
        institution=user.institution,
        is_active=user.is_active,
        created_at=user.created_at
    )


@router.put("/users/{user_id}/toggle-active")
def toggle_user_active(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """Toggle user active status (admin only)."""
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
    
    user.is_active = not user.is_active
    db.commit()
    
    # Log action
    log_action(db, admin.id, "toggle_user_active", f"User {user.email} {'activated' if user.is_active else 'deactivated'}")
    
    return {"message": f"User {'activated' if user.is_active else 'deactivated'}", "is_active": user.is_active}


@router.put("/users/{user_id}/role")
def update_user_role(
    user_id: str,
    role: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """Update user role (admin only)."""
    if role not in ["admin", "practitioner", "researcher"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    old_role = user.role
    user.role = role
    db.commit()
    
    log_action(db, admin.id, "update_user_role", f"User {user.email} role changed from {old_role} to {role}")
    
    return {"message": "Role updated", "role": role}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """Delete a user (admin only)."""
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    email = user.email
    db.delete(user)
    db.commit()
    
    log_action(db, admin.id, "delete_user", f"User {email} deleted")
    
    return {"message": "User deleted"}


def log_action(db: Session, user_id: str, action: str, details: str):
    """Log an admin action."""
    try:
        log = AuditLogDB(user_id=user_id, action=action, details=details)
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log action: {e}")


# Audit Logs Routes
@router.get("/logs")
def list_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    action: str = Query(None),
    user_id: str = Query(None),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """List audit logs (admin only)."""
    query = db.query(AuditLogDB).join(UserDB)

    if action:
        query = query.filter(AuditLogDB.action == action)
    if user_id:
        query = query.filter(AuditLogDB.user_id == user_id)

    logs = query.order_by(desc(AuditLogDB.created_at)).offset(skip).limit(limit).all()

    return [
        {
            "id": str(log.id),
            "user_id": log.user_id,
            "user_email": log.user.email if log.user else None,
            "user_name": log.user.full_name if log.user else None,
            "action": log.action,
            "details": log.details,
            "ip_address": log.ip_address,
            "created_at": log.created_at.isoformat() if log.created_at else None
        }
        for log in logs
    ]


@router.get("/stats")
def get_admin_stats(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    """Get platform statistics (admin only)."""
    from app.database import Patient as PatientDB, Document as DocumentDB, Analysis as AnalysisDB

    total_users = db.query(UserDB).count()
    active_users = db.query(UserDB).filter(UserDB.is_active == True).count()
    total_patients = db.query(PatientDB).count()
    total_documents = db.query(DocumentDB).count()
    total_analyses = db.query(AnalysisDB).count()

    # Recent activity
    from datetime import datetime, timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_analyses = db.query(AnalysisDB).filter(AnalysisDB.created_at >= week_ago).count()
    recent_users = db.query(UserDB).filter(UserDB.created_at >= week_ago).count()

    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_patients": total_patients,
        "total_documents": total_documents,
        "total_analyses": total_analyses,
        "recent_analyses_7d": recent_analyses,
        "new_users_7d": recent_users
    }

