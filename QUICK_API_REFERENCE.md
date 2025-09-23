# 🚀 Quick API Reference Card

## Most Common Endpoints

### 📄 **Get Recent Documents**

```bash
GET /api/documents?limit=20
```

### 🔍 **Search Documents**

```bash
GET /api/documents/search?q=John&limit=10
```

### 📋 **Get Document Details**

```bash
GET /api/documents/{document-id}
```

### 🚨 **Get All Alerts**

```bash
GET /api/alerts?limit=50&include_resolved=false
```

### ⚠️ **Get Urgent Alerts Only**

```bash
GET /api/alerts/urgent?limit=20
```

### 📊 **Get Statistics**

```bash
GET /api/statistics
```

### 🚀 **Process Document**

```bash
curl -X POST http://localhost:8000/api/extract-document \
  -F "document=@medical-report.docx"
```

### ✅ **Resolve Alert**

```bash
POST /api/alerts/{alert-id}/resolve?resolved_by=user123
```

---

## 📋 Parameter Quick Reference

| Endpoint               | Required Params   | Optional Params                          |
| ---------------------- | ----------------- | ---------------------------------------- |
| `/documents`           | -                 | `limit` (10)                             |
| `/documents/search`    | `q` (string)      | `limit` (20)                             |
| `/documents/{id}`      | `id` (UUID)       | -                                        |
| `/alerts`              | -                 | `limit` (50), `include_resolved` (false) |
| `/alerts/urgent`       | -                 | `limit` (20)                             |
| `/alerts/{id}`         | `id` (UUID)       | -                                        |
| `/alerts/{id}/resolve` | `id` (UUID)       | `resolved_by` (string)                   |
| `/extract-document`    | `document` (file) | -                                        |

---

## 🎯 Response Status Codes

- ✅ **200** - Success
- ❌ **400** - Bad Request
- 🔍 **404** - Not Found
- 🚫 **415** - Unsupported Media Type
- ⚠️ **422** - Validation Error
- 💥 **500** - Server Error
