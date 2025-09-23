# Healthcare Document Analysis API - Endpoint Specifications

## Complete API Reference with Payloads and Parameters

### Base URL: `http://localhost:8000/api`

---

## 📄 Document Management Endpoints

### 1. **GET** `/api/documents`

**Purpose**: Get recent documents with alert counts

**Parameters (Query String)**:

```
limit: integer (optional, default: 10)
  - Maximum number of documents to return
  - Range: 1-100
  - Example: ?limit=20
```

**Request Example**:

```bash
GET /api/documents?limit=15
```

**Response Payload**:

```json
[
  {
    "id": "ea1d8bab-3598-4cc3-bff7-e529a255347c",
    "originalName": "MRI-Report.docx",
    "fileSize": 7294,
    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "extractedText": "Report 4: MRI Diagnostic Report...",
    "pages": 1,
    "confidence": 0.9429,
    "entities": ["..."],
    "tables": ["..."],
    "formFields": ["..."],
    "patientName": "John Smith",
    "patientEmail": "john.smith@example.com",
    "claimNumber": "1234",
    "reportTitle": "MRI Diagnostic Report",
    "reportDate": "2025-09-10T00:00:00",
    "status": "normal",
    "summary": ["Patient shows moderate disc herniation", "Requires follow-up"],
    "originalReport": "Full report text...",
    "processingTimeMs": 19018,
    "analysisSuccess": true,
    "errorMessage": null,
    "createdAt": "2025-09-22T19:42:37.104",
    "updatedAt": "2025-09-22T19:42:37.104",
    "alert_count": 1,
    "urgent_alert_count": 0
  }
]
```

---

### 2. **GET** `/api/documents/{document_id}`

**Purpose**: Get complete details for a specific document

**Parameters (Path)**:

```
document_id: string (required)
  - Document UUID
  - Format: UUID v4
  - Example: ea1d8bab-3598-4cc3-bff7-e529a255347c
```

**Request Example**:

```bash
GET /api/documents/ea1d8bab-3598-4cc3-bff7-e529a255347c
```

**Response Payload**: Same as above but for single document

**Error Responses**:

```json
// 404 - Document not found
{
  "detail": "Document not found"
}

// 500 - Server error
{
  "detail": "Failed to retrieve document: [error message]"
}
```

---

### 3. **GET** `/api/documents/search`

**Purpose**: Search documents by patient name, claim number, report title, or filename

**Parameters (Query String)**:

```
q: string (required)
  - Search query string
  - Minimum length: 2 characters
  - Searches in: patientName, claimNumber, reportTitle, originalName
  - Case insensitive
  - Example: ?q=John

limit: integer (optional, default: 20)
  - Maximum number of results
  - Range: 1-100
  - Example: ?limit=10
```

**Request Examples**:

```bash
GET /api/documents/search?q=John&limit=10
GET /api/documents/search?q=MRI
GET /api/documents/search?q=CLM-2024
```

**Response Payload**: Array of documents (same format as `/api/documents`)

**Error Responses**:

```json
// 400 - Invalid query
{
  "detail": "Search query must be at least 2 characters long"
}
```

---

### 4. **GET** `/api/documents/{document_id}/alerts`

**Purpose**: Get all alerts for a specific document

**Parameters (Path)**:

```
document_id: string (required)
  - Document UUID
  - Example: ea1d8bab-3598-4cc3-bff7-e529a255347c
```

**Request Example**:

```bash
GET /api/documents/ea1d8bab-3598-4cc3-bff7-e529a255347c/alerts
```

**Response Payload**:

```json
[
  {
    "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "alertType": "Work Status Review",
    "title": "TTD Exceeding 45 Days",
    "date": "2025-09-22",
    "status": "normal",
    "description": null,
    "documentId": "ea1d8bab-3598-4cc3-bff7-e529a255347c",
    "isResolved": false,
    "resolvedAt": null,
    "resolvedBy": null,
    "createdAt": "2025-09-22T19:42:37.104",
    "updatedAt": "2025-09-22T19:42:37.104"
  }
]
```

---

## 🚨 Alert Management Endpoints

### 5. **GET** `/api/alerts`

**Purpose**: Get all alerts with optional filtering

**Parameters (Query String)**:

```
limit: integer (optional, default: 50)
  - Maximum number of alerts to return
  - Range: 1-200
  - Example: ?limit=100

include_resolved: boolean (optional, default: false)
  - Whether to include resolved alerts
  - Values: true, false
  - Example: ?include_resolved=true
```

**Request Examples**:

```bash
GET /api/alerts?limit=100&include_resolved=true
GET /api/alerts?limit=20
GET /api/alerts
```

**Response Payload**:

```json
[
  {
    "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "alertType": "Work Status Review",
    "title": "TTD Exceeding 45 Days",
    "date": "2025-09-22",
    "status": "urgent",
    "description": "Patient has been on TTD status for over 45 days",
    "documentId": "ea1d8bab-3598-4cc3-bff7-e529a255347c",
    "isResolved": false,
    "resolvedAt": null,
    "resolvedBy": null,
    "createdAt": "2025-09-22T19:42:37.104",
    "updatedAt": "2025-09-22T19:42:37.104",
    "patientName": "John Smith",
    "reportTitle": "MRI Diagnostic Report",
    "originalName": "MRI-Report.docx"
  }
]
```

---

### 6. **GET** `/api/alerts/urgent`

**Purpose**: Get only urgent unresolved alerts

**Parameters (Query String)**:

```
limit: integer (optional, default: 20)
  - Maximum number of alerts to return
  - Range: 1-100
  - Example: ?limit=10
```

**Request Example**:

```bash
GET /api/alerts/urgent?limit=10
```

**Response Payload**: Same format as `/api/alerts` but filtered for urgent status

---

### 7. **GET** `/api/alerts/{alert_id}`

**Purpose**: Get specific alert details with document context

**Parameters (Path)**:

```
alert_id: string (required)
  - Alert UUID
  - Example: f47ac10b-58cc-4372-a567-0e02b2c3d479
```

**Request Example**:

```bash
GET /api/alerts/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

**Response Payload**: Single alert object (same format as in `/api/alerts` array)

**Error Responses**:

```json
// 404 - Alert not found
{
  "detail": "Alert not found"
}
```

---

### 8. **POST** `/api/alerts/{alert_id}/resolve`

**Purpose**: Mark an alert as resolved

**Parameters (Path)**:

```
alert_id: string (required)
  - Alert UUID
  - Example: f47ac10b-58cc-4372-a567-0e02b2c3d479
```

**Parameters (Query String)**:

```
resolved_by: string (optional, default: "system")
  - User ID who resolved the alert
  - Example: ?resolved_by=user123
```

**Request Example**:

```bash
POST /api/alerts/f47ac10b-58cc-4372-a567-0e02b2c3d479/resolve?resolved_by=user123
```

**Request Body**: None required

**Response Payload**:

```json
{
  "message": "Alert resolved successfully",
  "alert_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```

**Error Responses**:

```json
// 404 - Alert not found
{
  "detail": "Alert not found or could not be resolved"
}
```

---

## 🚀 Document Processing Endpoint

### 9. **POST** `/api/extract-document`

**Purpose**: Upload and process documents with AI analysis

**Parameters**: None (multipart form data)

**Request Body (Multipart Form Data)**:

```
document: file (required)
  - File to be processed
  - Supported formats: PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF, GIF
  - Maximum size: 40MB (configurable)
  - Content-Type: multipart/form-data
```

**Request Example (cURL)**:

```bash
curl -X POST \
  http://localhost:8000/api/extract-document \
  -F "document=@/path/to/medical-report.docx"
```

**Request Example (JavaScript)**:

```javascript
const formData = new FormData();
formData.append("document", fileInput.files[0]);

fetch("/api/extract-document", {
  method: "POST",
  body: formData,
});
```

**Response Payload**:

```json
{
  "text": "Report 4: MRI Diagnostic Report...",
  "pages": 1,
  "entities": [
    {
      "type": "PERSON",
      "mentionText": "John Smith",
      "confidence": 0.98
    }
  ],
  "tables": [
    {
      "headers": ["Date", "Status"],
      "rows": [["2025-09-10", "Assessment"]]
    }
  ],
  "formFields": [
    {
      "fieldName": "Patient Name",
      "fieldValue": "John Smith"
    }
  ],
  "confidence": 0.9429,
  "success": true,
  "error": null,
  "fileInfo": {
    "originalName": "MRI-Report.docx",
    "fileSize": 7294,
    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  },
  "summary": "Legacy field - see comprehensive_analysis",
  "comprehensive_analysis": {
    "original_report": "Full report text...",
    "report_json": {
      "patient_name": "John Smith",
      "patient_email": "john.smith@example.com",
      "claim_no": "1234",
      "report_title": "MRI Diagnostic Report",
      "time_day": "2025-09-10T00:00:00Z",
      "status": "normal"
    },
    "summary": [
      "Patient shows moderate disc herniation at L4-L5",
      "Requires follow-up imaging in 6 months"
    ],
    "work_status_alert": [
      {
        "alert_type": "Work Status Review",
        "title": "TTD Status Review Required",
        "date": "2025-09-22",
        "status": "normal"
      }
    ]
  },
  "database_id": "ea1d8bab-3598-4cc3-bff7-e529a255347c",
  "processing_time_ms": 19018
}
```

**Error Responses**:

```json
// 400 - Invalid file
{
  "detail": "File size exceeds maximum limit of 40MB"
}

// 415 - Unsupported file type
{
  "detail": "Unsupported file type. Please upload PDF, DOCX, or image files."
}

// 500 - Processing error
{
  "detail": "Document processing failed: [error details]"
}
```

---

## 📊 Analytics & Information Endpoints

### 10. **GET** `/api/statistics`

**Purpose**: Get database statistics for dashboards

**Parameters**: None

**Request Example**:

```bash
GET /api/statistics
```

**Response Payload**:

```json
{
  "total_documents": 25,
  "urgent_alerts": 3,
  "resolved_alerts": 12,
  "recent_documents": 8,
  "last_updated": "2025-09-22T19:45:30.123456",
  "processing_stats": {
    "avg_processing_time_ms": 15000,
    "success_rate": 0.96
  },
  "alert_breakdown": {
    "urgent": 3,
    "normal": 8,
    "low": 2
  }
}
```

---

### 11. **GET** `/api/api-info`

**Purpose**: Get comprehensive API documentation

**Parameters**: None

**Request Example**:

```bash
GET /api/api-info
```

**Response Payload**:

```json
{
  "title": "Healthcare Document Analysis API",
  "version": "1.0.0",
  "base_url": "/api",
  "endpoints": {
    "document_processing": {
      "POST /api/extract-document": {
        "description": "Upload and process document with Document AI and GPT-4o analysis",
        "parameters": ["document (file)"],
        "returns": "ExtractionResult with comprehensive analysis"
      }
    },
    "document_management": {
      "GET /api/documents": {
        "description": "Get recent documents with alert counts",
        "parameters": ["limit (default: 10)"],
        "returns": "List of recent documents"
      }
      // ... more endpoints
    }
  },
  "total_endpoints": 11,
  "features": [
    "Document processing with Google Document AI",
    "AI-powered analysis with GPT-4o",
    "Database persistence with PostgreSQL",
    "Automated alert generation"
  ]
}
```

---

## 🔧 Common Response Headers

All endpoints return these common headers:

```
Content-Type: application/json
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
```

## 🚨 Error Response Format

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (resource doesn't exist)
- `415` - Unsupported Media Type (invalid file type)
- `422` - Validation Error (invalid data format)
- `500` - Internal Server Error

## 🧪 Testing Examples

You can test all endpoints using the provided test script:

```bash
python test_api_endpoints.py
```

Or use tools like Postman, curl, or any HTTP client with the specifications above.
