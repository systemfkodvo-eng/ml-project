"""
Database initialization script.
Creates indexes and initial admin user.
"""
import asyncio
import sys
sys.path.insert(0, '.')

from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from datetime import datetime

from app.config import get_settings
from app.database import (
    USERS_COLLECTION, 
    PATIENTS_COLLECTION, 
    DOCUMENTS_COLLECTION,
    EXTRACTIONS_COLLECTION,
    ANALYSES_COLLECTION
)

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def init_database():
    """Initialize database with indexes and optional seed data."""
    print("🔧 Initializing database...")
    
    client = AsyncIOMotorClient(settings.mongodb_url)
    db = client[settings.database_name]
    
    # Create indexes for Users
    print("📝 Creating indexes for users...")
    await db[USERS_COLLECTION].create_index("email", unique=True)
    
    # Create indexes for Patients
    print("📝 Creating indexes for patients...")
    await db[PATIENTS_COLLECTION].create_index([
        ("practitioner_id", 1),
        ("patient_code", 1)
    ], unique=True)
    await db[PATIENTS_COLLECTION].create_index("created_at")
    
    # Create indexes for Documents
    print("📝 Creating indexes for documents...")
    await db[DOCUMENTS_COLLECTION].create_index("patient_id")
    await db[DOCUMENTS_COLLECTION].create_index("practitioner_id")
    await db[DOCUMENTS_COLLECTION].create_index("status")
    await db[DOCUMENTS_COLLECTION].create_index("created_at")
    
    # Create indexes for Extractions
    print("📝 Creating indexes for extractions...")
    await db[EXTRACTIONS_COLLECTION].create_index("document_id", unique=True)
    await db[EXTRACTIONS_COLLECTION].create_index("patient_id")
    await db[EXTRACTIONS_COLLECTION].create_index("status")
    
    # Create indexes for Analyses
    print("📝 Creating indexes for analyses...")
    await db[ANALYSES_COLLECTION].create_index("patient_id")
    await db[ANALYSES_COLLECTION].create_index("practitioner_id")
    await db[ANALYSES_COLLECTION].create_index("created_at")
    
    # Create default admin user if not exists
    print("👤 Checking for admin user...")
    admin = await db[USERS_COLLECTION].find_one({"email": "admin@healthrisk.com"})
    
    if not admin:
        print("👤 Creating default admin user...")
        admin_user = {
            "email": "admin@healthrisk.com",
            "full_name": "System Administrator",
            "role": "admin",
            "specialty": None,
            "institution": "Health Risk Platform",
            "hashed_password": pwd_context.hash("admin123"),  # Change in production!
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        await db[USERS_COLLECTION].insert_one(admin_user)
        print("✅ Admin user created (email: admin@healthrisk.com, password: admin123)")
        print("⚠️  IMPORTANT: Change the admin password in production!")
    else:
        print("✅ Admin user already exists")
    
    client.close()
    print("✅ Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(init_database())

