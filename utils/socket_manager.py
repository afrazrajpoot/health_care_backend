# utils/socket_manager.py (simplified: no RedisManager needed now, since emits are in main process)
import socketio

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[
        "*",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
)

@sio.event
async def connect(sid, environ, auth=None):
    print(f"🔌 Client connected: {sid}")
    if auth:
        print(f"   📝 Auth data: {auth}")
        user_id = auth.get('userId')
        if user_id:
            await sio.enter_room(sid, f"user_{user_id}")
            print(f"   📥 Joined room: user_{user_id}")
    return True  # Accept connection

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")