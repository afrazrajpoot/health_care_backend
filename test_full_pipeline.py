#!/usr/bin/env python3
"""
Full Pipeline Test - Document Analysis to Database
Tests the complete flow: Document Processing -> AI Analysis -> Database Storage
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.database_service import DatabaseService
from services.report_analyzer import ReportAnalyzer
from models.schemas import ExtractionResult, ComprehensiveAnalysis, PatientInfo, WorkStatusAlert

async def test_full_pipeline():
    """Test the complete document analysis to database pipeline"""
    
    print("🚀 Starting Full Pipeline Test")
    print("=" * 50)
    
    # Initialize services
    db_service = DatabaseService()
    await db_service.connect()
    print("✅ Database service connected")
    
    # Create mock extraction result with comprehensive analysis
    mock_text = """
    MEDICAL REPORT
    
    Patient Name: John Doe
    Patient Email: john.doe@email.com
    Claim Number: CLM-2024-001
    Report Title: Work Capacity Assessment
    Report Date: 2024-01-15
    Status: urgent
    
    Summary:
    - Patient shows significant improvement in mobility
    - Recommended return to light duties within 2 weeks
    - Follow-up appointment scheduled
    
    Work Status Assessment:
    Patient is currently unable to perform heavy lifting duties.
    Recommendation: Light duties only for the next 4 weeks.
    Next review: 2024-02-15
    
    Medical Findings:
    - No acute distress
    - Range of motion improved by 40%
    - Patient reports reduced pain levels
    """
    
    # Mock extraction result
    extraction_result = ExtractionResult(
        text=mock_text,
        pages=2,
        confidence=0.95,
        entities=[
            {"type": "PERSON", "mentionText": "John Doe", "confidence": 0.98},
            {"type": "DATE", "mentionText": "2024-01-15", "confidence": 0.95}
        ],
        tables=[
            {"headers": ["Date", "Status"], "rows": [["2024-01-15", "Assessment"]]}
        ],
        formFields=[
            {"fieldName": "Patient Name", "fieldValue": "John Doe"},
            {"fieldName": "Claim Number", "fieldValue": "CLM-2024-001"}
        ],
        success=True,
        error=None,
        comprehensive_analysis=None  # Will be added by analyzer
    )
    
    print("📝 Mock extraction result created")
    
    # Analyze the document using ReportAnalyzer
    try:
        analyzer = ReportAnalyzer()
        print("🤖 Starting AI analysis...")
        
        # Perform comprehensive analysis
        analysis_result = await analyzer.analyze_document(mock_text)
        
        if analysis_result:
            extraction_result.comprehensive_analysis = analysis_result
            print("✅ AI analysis completed successfully")
            print(f"📊 Analysis summary: {len(analysis_result.summary) if analysis_result.summary else 0} points")
            print(f"🚨 Alerts generated: {len(analysis_result.work_status_alert) if analysis_result.work_status_alert else 0}")
        else:
            print("❌ AI analysis failed - creating mock analysis")
            # Create mock comprehensive analysis for testing
            mock_analysis = ComprehensiveAnalysis(
                report_json=PatientInfo(
                    patient_name="John Doe",
                    patient_email="john.doe@email.com",
                    claim_no="CLM-2024-001",
                    report_title="Work Capacity Assessment",
                    time_day="2024-01-15T00:00:00Z",
                    status="urgent"
                ),
                summary=[
                    "Patient shows significant improvement in mobility",
                    "Recommended return to light duties within 2 weeks",
                    "Follow-up appointment scheduled"
                ],
                work_status_alert=[
                    WorkStatusAlert(
                        alert_type="Work Status Review",
                        title="Light Duties Recommended",
                        date="2024-02-15",
                        status="urgent"
                    )
                ],
                original_report=mock_text
            )
            extraction_result.comprehensive_analysis = mock_analysis
            print("✅ Mock analysis created for testing")
            
    except Exception as e:
        print(f"⚠️ AI analysis failed: {str(e)}")
        print("📝 Creating minimal mock analysis for database testing...")
        
        # Create minimal mock analysis
        mock_analysis = ComprehensiveAnalysis(
            report_json=PatientInfo(
                patient_name="John Doe",
                patient_email="john.doe@email.com", 
                claim_no="CLM-2024-001",
                report_title="Work Capacity Assessment",
                time_day="2024-01-15T00:00:00Z",
                status="urgent"
            ),
            summary=["Test summary point"],
            work_status_alert=[
                WorkStatusAlert(
                    alert_type="Work Status Review",
                    title="Test Alert",
                    date="2024-02-15",
                    status="urgent"
                )
            ],
            original_report=mock_text
        )
        extraction_result.comprehensive_analysis = mock_analysis
    
    # Save to database
    print("💾 Saving analysis results to database...")
    try:
        document_id = await db_service.save_document_analysis(
            extraction_result=extraction_result,
            file_name="test_document.pdf",
            file_size=1024000,
            mime_type="application/pdf",
            processing_time_ms=5000
        )
        
        print(f"✅ Document saved successfully with ID: {document_id}")
        
        # Verify the saved data
        print("🔍 Verifying saved data...")
        document = await db_service.get_document(document_id)
        
        if document:
            print("✅ Document retrieved successfully")
            print(f"📄 Patient: {document.get('patientName')}")
            print(f"📋 Claim: {document.get('claimNumber')}")
            print(f"📅 Report Date: {document.get('reportDate')}")
            print(f"🎯 Status: {document.get('status')}")
            print(f"📝 Summary Points: {len(document.get('summary', []))}")
            
            # Check alerts
            alerts = await db_service.get_document_alerts(document_id)
            print(f"🚨 Alerts created: {len(alerts)}")
            for alert in alerts:
                print(f"   ⚠️ {alert.get('title')} - {alert.get('status')}")
                
        else:
            print("❌ Failed to retrieve saved document")
            
    except Exception as e:
        print(f"❌ Database save failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await db_service.disconnect()
        print("🔒 Database connection closed")
    
    print("\n" + "=" * 50)
    print("🏁 Full Pipeline Test Complete")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())