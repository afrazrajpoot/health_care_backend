# utils/socket_manager.py (updated)
import socketio

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[
        "*",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    ping_timeout=60,
    ping_interval=25
)

@sio.event
async def connect(sid, environ, auth=None):
    print(f"🔌 Client connected: {sid}")
    print(f"   📝 Auth data: {auth}")
    
    if auth and 'userId' in auth:
        user_id = auth.get('userId')
        if user_id:
            await sio.enter_room(sid, f"user_{user_id}")
            await sio.save_session(sid, {'user_id': user_id})
            print(f"   📥 Joined room: user_{user_id}")
            return True  # Accept connection
        else:
            print("   ⚠️ No user ID in auth")
            return False  # Reject connection
    else:
        print("   ⚠️ No auth data provided")
        # For development, allow connection without auth
        # In production, you might want to reject this
        return True

@sio.event
async def disconnect(sid):
    session = await sio.get_session(sid)
    user_id = session.get('user_id') if session else None
    print(f"🔌 Client disconnected: {sid} (user: {user_id})")
    if user_id:
        await sio.leave_room(sid, f"user_{user_id}")

@sio.event
async def join(sid, data):
    """Allow clients to join user-specific rooms"""
    print(f"📥 Join request from {sid}: {data}")
    if isinstance(data, str) and data.startswith('user_'):
        user_id = data.replace('user_', '')
        await sio.enter_room(sid, data)
        await sio.save_session(sid, {'user_id': user_id})
        print(f"   ✅ Client {sid} joined room: {data}")
        return {'status': 'joined', 'room': data}
    else:
        print(f"   ⚠️ Invalid room join attempt: {data}")
        return {'status': 'error', 'message': 'Invalid room format'}

# Add error handling
@sio.event
async def connect_error(data):
    print(f"❌ Socket connection error: {data}")

@sio.event
async def error(data):
    print(f"❌ Socket error: {data}")