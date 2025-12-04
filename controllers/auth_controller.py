from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from prisma import Prisma
import os
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import Optional
import uuid  # Added missing import
import re    # Added for password validation

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secure-jwt-secret-key-change-in-production")
SYNC_SECRET = os.getenv("SYNC_SECRET", "your-sync-secret-key-for-nextauth")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Pydantic Models
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class SyncLoginRequest(BaseModel):
    user_id: str
    user_email: str
    sync_token: str

class UserSignUp(BaseModel):
    email: EmailStr
    password: str
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phoneNumber: Optional[str] = None
    role: Optional[str] = "patient"  # Default role

class UserResponse(BaseModel):
    id: str
    email: Optional[EmailStr] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phoneNumber: Optional[str] = None
    role: Optional[str] = None
    physicianId: Optional[str] = None
    emailVerified: Optional[datetime] = None
    image: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    user_id: Optional[str] = None

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Password complexity validation function
def validate_password_complexity(password: str) -> tuple[bool, str]:
    """Validate password complexity"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is valid"

# Database dependency
async def get_db():
    """Database connection dependency"""
    db = Prisma()
    await db.connect()
    try:
        yield db
    finally:
        await db.disconnect()

# Pure async function for token verification (no Depends)
async def verify_token(token: str, db: Prisma) -> UserResponse:
    """Verify JWT token and return user - for manual calls"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    user = await db.user.find_unique(where={"id": token_data.user_id})
    
    if user is None:
        raise credentials_exception
    
    return UserResponse(
        id=user.id,
        email=user.email,
        firstName=user.firstName,
        lastName=user.lastName,
        phoneNumber=user.phoneNumber,
        role=user.role,
        physicianId=user.physicianId,
        emailVerified=user.emailVerified,
        image=user.image,
        createdAt=user.createdAt,
        updatedAt=user.updatedAt
    )

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: Prisma = Depends(get_db)  # Fixed: using proper dependency
) -> UserResponse:
    """Dependency version - for route dependencies"""
    return await verify_token(token, db)

async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)):
    if current_user.role == "inactive":
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# SYNC LOGIN ENDPOINT - For NextAuth automatic login
@router.post("/sync-login", response_model=Token)
async def sync_login(login_data: SyncLoginRequest):
    """Automatic login from NextAuth - called during NextAuth login process"""
    print(f"🔄 Sync login attempt for user: {login_data.user_email}")
    
    try:
        # Verify sync token (shared secret between NextAuth and FastAPI)
        if login_data.sync_token != SYNC_SECRET:
            print(f"❌ Sync token mismatch! Received: '{login_data.sync_token}', Expected: '{SYNC_SECRET}'")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid sync token"
            )
        
        print("✅ Sync token validated successfully")
        
        # Verify user exists in database
        db = Prisma()
        await db.connect()
        
        try:
            user = await db.user.find_unique(where={"id": login_data.user_id})
            if not user:
                print(f"❌ User not found: {login_data.user_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found in database"
                )
            
            # ✅ PRINT USER DETAILS ON SYNC LOGIN
            print("=" * 50)
            print("🔄 SYNC LOGIN SUCCESSFUL:")
            print(f"   👤 User ID: {user.id}")
            print(f"   📧 Email: {user.email}")
            print(f"   🏥 Physician ID: {user.physicianId}")
            print(f"   👨‍⚕️ Name: {user.firstName} {user.lastName}")
            print(f"   📞 Phone: {user.phoneNumber}")
            print(f"   🎯 Role: {user.role}")
            print(f"   📅 Created: {user.createdAt}")
            print("=" * 50)
            
            # Verify email matches (additional security)
            if user.email != login_data.user_email:
                print(f"❌ Email mismatch! DB: {user.email}, Request: {login_data.user_email}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User email mismatch"
                )
            
            # Create FastAPI JWT token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user.id}, expires_delta=access_token_expires
            )
            
            print(f"✅ FastAPI token created for user: {user.email}")
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": UserResponse(
                    id=user.id,
                    email=user.email,
                    firstName=user.firstName,
                    lastName=user.lastName,
                    phoneNumber=user.phoneNumber,
                    role=user.role,
                    physicianId=user.physicianId,
                    emailVerified=user.emailVerified,
                    image=user.image,
                    createdAt=user.createdAt,
                    updatedAt=user.updatedAt
                )
            }
        finally:
            await db.disconnect()
            
    except HTTPException as e:
        print(f"❌ HTTP Exception in sync login: {e.detail}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error in sync login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync login failed: {str(e)}"
        )

# Regular login endpoint
@router.post("/login", response_model=Token)
async def login(login_data: UserLogin):
    """Login user and return JWT token"""
    db = Prisma()
    await db.connect()
    
    try:
        # Find user by email
        user = await db.user.find_unique(where={"email": login_data.email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if user has password (might be OAuth user)
        if not user.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Please use social login or reset your password"
            )
        
        # Verify password
        if not verify_password(login_data.password, user.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # ✅ PRINT USER DETAILS ON REGULAR LOGIN
        print("=" * 50)
        print("🔐 REGULAR LOGIN SUCCESSFUL:")
        print(f"   👤 User ID: {user.id}")
        print(f"   📧 Email: {user.email}")
        print(f"   🏥 Physician ID: {user.physicianId}")
        print(f"   👨‍⚕️ Name: {user.firstName} {user.lastName}")
        print(f"   📞 Phone: {user.phoneNumber}")
        print(f"   🎯 Role: {user.role}")
        print(f"   📅 Created: {user.createdAt}")
        print("=" * 50)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.id}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse(
                id=user.id,
                email=user.email,
                firstName=user.firstName,
                lastName=user.lastName,
                phoneNumber=user.phoneNumber,
                role=user.role,
                physicianId=user.physicianId,
                emailVerified=user.emailVerified,
                image=user.image,
                createdAt=user.createdAt,
                updatedAt=user.updatedAt
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )
    finally:
        await db.disconnect()

# Update the signup endpoint
@router.post("/signup", response_model=Token)
async def sign_up(signup_data: UserSignUp):
    """Register a new user - compatible with NextAuth schema"""
    db = Prisma()
    await db.connect()
    
    try:
        # Check if user already exists
        existing_user = await db.user.find_unique(where={"email": signup_data.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Validate password complexity
        is_valid, message = validate_password_complexity(signup_data.password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        
        # Hash password
        hashed_password = pwd_context.hash(signup_data.password)
        
        # Validate role
        valid_roles = [ "Physician", "Staff"]
        user_role = signup_data.role if signup_data.role in valid_roles else "patient"
        
        # Create user with proper NextAuth schema fields
        user = await db.user.create(
            data={
                "id": str(uuid.uuid4()),  # Generate UUID
                "email": signup_data.email,
                "password": hashed_password,
                "firstName": signup_data.firstName,
                "lastName": signup_data.lastName,
                "phoneNumber": signup_data.phoneNumber,
                "role": user_role,
                "emailVerified": None,  # Will be set when email is verified
                "image": None,  # Can be updated later
                "physicianId": None,  # Will be set for physician users
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
            }
        )
        
        # ✅ PRINT USER DETAILS ON SIGNUP
        print("=" * 50)
        print("🎉 USER SIGNUP SUCCESSFUL (NextAuth Schema):")
        print(f"   👤 User ID: {user.id}")
        print(f"   📧 Email: {user.email}")
        print(f"   🏥 Physician ID: {user.physicianId}")
        print(f"   👨‍⚕️ Name: {user.firstName} {user.lastName}")
        print(f"   📞 Phone: {user.phoneNumber}")
        print(f"   🎯 Role: {user.role}")
        print(f"   📅 Email Verified: {user.emailVerified}")
        print(f"   🖼️ Image: {user.image}")
        print(f"   📅 Created: {user.createdAt}")
        print("=" * 50)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.id}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse(
                id=user.id,
                email=user.email,
                firstName=user.firstName,
                lastName=user.lastName,
                phoneNumber=user.phoneNumber,
                role=user.role,
                physicianId=user.physicianId,
                emailVerified=user.emailVerified,
                image=user.image,
                createdAt=user.createdAt,
                updatedAt=user.updatedAt
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signup failed: {str(e)}"
        )
    finally:
        await db.disconnect()

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserResponse = Depends(get_current_active_user)):
    """Get current user profile"""
    return current_user

@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: UserResponse = Depends(get_current_active_user)):
    """Refresh JWT token"""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.id}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": current_user
    }