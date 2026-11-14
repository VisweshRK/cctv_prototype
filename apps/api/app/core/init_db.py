"""Database initialization and seeding script.
Ensures tables and default super admin and demo accounts are created immediately.
"""

import logging
from sqlalchemy import select

from apps.api.app.core.database import SessionLocal, Base, engine
from apps.api.app.core.security import hash_password
from apps.api.app.models.entities import Camera, Household, User

logger = logging.getLogger(__name__)


def init_database():
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        # Check if super admin exists
        stmt = select(User).where(User.email == "admin@presence.ai")
        user = db.execute(stmt).scalar_one_or_none()

        if not user:
            admin_household = Household(name="Presence Corporate Super Admin")
            db.add(admin_household)
            db.flush()

            super_admin = User(
                email="admin@presence.ai",
                hashed_password=hash_password("admin123"),
                full_name="Presence Super Administrator",
                role="SUPER_ADMIN",
                household_id=admin_household.id
            )
            db.add(super_admin)

            demo_household = Household(name="Sharma Residence")
            db.add(demo_household)
            db.flush()

            demo_user = User(
                email="demo@presence.ai",
                hashed_password=hash_password("demo123"),
                full_name="Rajesh Sharma",
                role="HOUSEHOLD_ADMIN",
                household_id=demo_household.id
            )
            db.add(demo_user)

            demo_cam = Camera(
                id="camera_main",
                household_id=demo_household.id,
                name="Living Room Main Stream",
                stream_type="synthetic",
                location="Living Room",
                is_online=True,
                capabilities={"audio": True, "ptz": False, "night_vision": True, "fps": 15}
            )
            db.add(demo_cam)
            db.commit()
            print("Database initialized and seeded.")
        else:
            print("Database already initialized.")


if __name__ == "__main__":
    init_database()
