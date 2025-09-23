#!/usr/bin/env python3
"""
Test database integration for document analysis system
"""

import asyncio
import logging
from services.database_service import DatabaseService

async def test_database_connection():
    """Test basic database connectivity"""
    print("🔧 Testing Database Connection")
    print("=" * 50)
    
    db_service = None
    try:
        # Initialize and connect
        print("📡 Connecting to database...")
        db_service = DatabaseService()
        await db_service.connect()
        print("✅ Database connection successful!")
        
        # Test statistics query
        print("\n📊 Testing database statistics...")
        stats = await db_service.get_statistics()
        print("✅ Statistics retrieved successfully:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test recent documents query
        print("\n📋 Testing recent documents query...")
        documents = await db_service.get_recent_documents(limit=5)
        print(f"✅ Found {len(documents)} documents")
        
        # Test urgent alerts query
        print("\n🚨 Testing urgent alerts query...")
        alerts = await db_service.get_urgent_alerts(limit=5)
        print(f"✅ Found {len(alerts)} urgent alerts")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {str(e)}")
        return False
        
    finally:
        if db_service:
            await db_service.disconnect()

async def main():
    """Run database tests"""
    print("🧪 DATABASE INTEGRATION TEST")
    print("=" * 60)
    
    # Configure logging to reduce noise
    logging.getLogger("document_ai").setLevel(logging.WARNING)
    
    success = await test_database_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 DATABASE INTEGRATION TEST PASSED!")
        print("✅ Your system is ready to store document analysis results.")
    else:
        print("❌ DATABASE INTEGRATION TEST FAILED!")
        print("Please check your database configuration and connection.")

if __name__ == "__main__":
    asyncio.run(main())