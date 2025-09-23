#!/usr/bin/env python3
"""
API Endpoints Test Script
Tests all the document and alert management endpoints
"""

import asyncio
import aiohttp
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api"

async def test_endpoints():
    """Test all API endpoints"""
    
    print("🚀 Testing Healthcare Document Analysis API Endpoints")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # Test API info endpoint
        print("\n1. 📋 Testing API Info Endpoint")
        try:
            async with session.get(f"{BASE_URL}/api-info") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ API Info: {data['title']} v{data['version']}")
                    print(f"📊 Total endpoints: {data['total_endpoints']}")
                else:
                    print(f"❌ API Info failed: {response.status}")
        except Exception as e:
            print(f"❌ API Info error: {e}")
        
        # Test statistics endpoint
        print("\n2. 📊 Testing Statistics Endpoint")
        try:
            async with session.get(f"{BASE_URL}/statistics") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Statistics: {data.get('total_documents', 0)} documents, {data.get('urgent_alerts', 0)} urgent alerts")
                else:
                    print(f"❌ Statistics failed: {response.status}")
        except Exception as e:
            print(f"❌ Statistics error: {e}")
        
        # Test documents endpoint
        print("\n3. 📄 Testing Documents Endpoint")
        try:
            async with session.get(f"{BASE_URL}/documents?limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Documents: Retrieved {len(data)} documents")
                    if data:
                        doc = data[0]
                        print(f"   📋 Sample: {doc.get('patientName', 'N/A')} - {doc.get('reportTitle', 'N/A')}")
                        
                        # Test specific document endpoint with the first document
                        doc_id = doc.get('id')
                        if doc_id:
                            print(f"\n4. 🔍 Testing Specific Document Endpoint")
                            async with session.get(f"{BASE_URL}/documents/{doc_id}") as doc_response:
                                if doc_response.status == 200:
                                    doc_data = await doc_response.json()
                                    print(f"✅ Document details: {doc_data.get('patientName', 'N/A')}")
                                else:
                                    print(f"❌ Document details failed: {doc_response.status}")
                            
                            # Test document alerts endpoint
                            print(f"\n5. 🚨 Testing Document Alerts Endpoint")
                            async with session.get(f"{BASE_URL}/documents/{doc_id}/alerts") as alerts_response:
                                if alerts_response.status == 200:
                                    alerts_data = await alerts_response.json()
                                    print(f"✅ Document alerts: {len(alerts_data)} alerts found")
                                else:
                                    print(f"❌ Document alerts failed: {alerts_response.status}")
                    else:
                        print("   📝 No documents found in database")
                else:
                    print(f"❌ Documents failed: {response.status}")
        except Exception as e:
            print(f"❌ Documents error: {e}")
        
        # Test all alerts endpoint
        print("\n6. 🚨 Testing All Alerts Endpoint")
        try:
            async with session.get(f"{BASE_URL}/alerts?limit=10") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ All alerts: Retrieved {len(data)} alerts")
                    if data:
                        alert = data[0]
                        print(f"   📋 Sample: {alert.get('title', 'N/A')} - {alert.get('status', 'N/A')}")
                        
                        # Test specific alert endpoint with the first alert
                        alert_id = alert.get('id')
                        if alert_id:
                            print(f"\n7. 🔍 Testing Specific Alert Endpoint")
                            async with session.get(f"{BASE_URL}/alerts/{alert_id}") as alert_response:
                                if alert_response.status == 200:
                                    alert_data = await alert_response.json()
                                    print(f"✅ Alert details: {alert_data.get('title', 'N/A')}")
                                else:
                                    print(f"❌ Alert details failed: {alert_response.status}")
                    else:
                        print("   📝 No alerts found in database")
                else:
                    print(f"❌ All alerts failed: {response.status}")
        except Exception as e:
            print(f"❌ All alerts error: {e}")
        
        # Test urgent alerts endpoint
        print("\n8. ⚠️ Testing Urgent Alerts Endpoint")
        try:
            async with session.get(f"{BASE_URL}/alerts/urgent?limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Urgent alerts: Retrieved {len(data)} urgent alerts")
                else:
                    print(f"❌ Urgent alerts failed: {response.status}")
        except Exception as e:
            print(f"❌ Urgent alerts error: {e}")
        
        # Test document search endpoint
        print("\n9. 🔍 Testing Document Search Endpoint")
        try:
            async with session.get(f"{BASE_URL}/documents/search?q=John&limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Document search: Found {len(data)} results for 'John'")
                else:
                    print(f"❌ Document search failed: {response.status}")
        except Exception as e:
            print(f"❌ Document search error: {e}")
        
    print("\n" + "=" * 60)
    print("🏁 API Endpoints Test Complete")
    print("\n📚 Available Endpoints Summary:")
    print("   📄 Documents: /api/documents, /api/documents/{id}, /api/documents/search")
    print("   🚨 Alerts: /api/alerts, /api/alerts/urgent, /api/alerts/{id}")
    print("   🔧 Management: /api/alerts/{id}/resolve")
    print("   📊 Analytics: /api/statistics, /api/api-info")
    print("   🚀 Processing: /api/extract-document")

if __name__ == "__main__":
    asyncio.run(test_endpoints())