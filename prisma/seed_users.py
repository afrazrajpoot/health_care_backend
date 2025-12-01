

import asyncio
from datetime import datetime
from prisma import Prisma
import bcrypt


USERS = [
    {
        "id": "ea0f1a1b-0891-49f2-9198-e6c687c94133",
        "firstName": "Kevin",
        "lastName": "Calhoun",
        "phoneNumber": "+1 (953) 739-7046",
        "email": "admin@kebilo.com",
        "password": "Pa$$w0rd!",  # Plaintext, will be hashed
        "role": "Physician",
        "createdAt": datetime.fromisoformat("2025-10-23 07:39:21.254"),
        "updatedAt": datetime.fromisoformat("2025-10-23 07:39:21.254"),
        "emailVerified": None,
        "image": None,
        "physicianId": None,
    },
    {
        "id": "fb413e50-3eab-4a03-be38-70cb44f051f1",
        "firstName": "Staff",
        "lastName": "User",
        "phoneNumber": "+1 (252) 379-8018",
        "email": "staff@kebilo.com",
        "password": "Pa$$w0rd!",  # Plaintext, will be hashed
        "role": "Staff",
        "createdAt": datetime.fromisoformat("2025-10-23 07:40:25.734"),
        "updatedAt": datetime.fromisoformat("2025-10-23 07:40:25.734"),
        "emailVerified": None,
        "image": None,
        "physicianId": "ea0f1a1b-0891-49f2-9198-e6c687c94133",
    },
]

async def main():
    prisma = Prisma()
    await prisma.connect()
    for user in USERS:
        # Hash the password before saving
        password = user["password"]
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        user_to_save = user.copy()
        user_to_save["password"] = hashed.decode("utf-8")
        try:
            await prisma.user.create(data=user_to_save)
            print(f"✅ Created user: {user['email']}")
        except Exception as e:
            print(f"⚠️  Could not create user {user['email']}: {e}")
    await prisma.disconnect()
    print("✅ Seeded users.")

if __name__ == "__main__":
    asyncio.run(main())
