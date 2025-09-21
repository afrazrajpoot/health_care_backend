import pandas as pd

# ✅ Only new jobs
new_jobs = [
    {"title": "AI Engineer", "description": "Work with machine learning models", "location": "Remote", "salary": 120000, "type": "FULL_TIME"},
    {"title": "DevOps Engineer", "description": "Maintain CI/CD pipelines", "location": "Karachi, PK", "salary": 80000, "type": "FULL_TIME"},
    {"title": "Product Manager", "description": "Lead product development", "location": "Lahore, PK", "salary": 100000, "type": "FULL_TIME"},
    {"title": "QA Tester", "description": "Test software applications", "location": "Islamabad, PK", "salary": 50000, "type": "CONTRACT"},
]

df = pd.DataFrame(new_jobs)

# Save only new jobs to ODS
df.to_excel("jobs1.ods", engine="odf", index=False)
print("jobs.ods created with only new jobs!")

# import pandas as pd

# # Employee data
# employee_data = [
#     {"firstName": "john", "lastName": "Khan", "email": "johnn@example.com", "phoneNumber": "03001234567", "position": "Frontend Developer", "salary": 50000, "department": "IT"},
#     {"firstName": "don", "lastName": "Ahmed", "email": "don@example.com", "phoneNumber": "03007654321", "position": "Backend Developer", "salary": 60000, "department": "IT"},
#     {"firstName": "Akhlaq", "lastName": "Hussain", "email": "akhlaq@example.com", "phoneNumber": "03009876543", "position": "Data Analyst", "salary": 45000, "department": "Analytics"},
#     {"firstName": "zeenat", "lastName": "Ali", "email": "zeenat.ali@example.com", "phoneNumber": "03001112233", "position": "UX Designer", "salary": 40000, "department": "Design"},
#     {"firstName": "Billa", "lastName": "Raza", "email": "bila.raza@example.com", "phoneNumber": "03005556677", "position": "Intern Developer", "salary": 0, "department": "IT"},
# ]

# # Create DataFrame
# df_employees = pd.DataFrame(employee_data)

# # Save as ODS file
# df_employees.to_excel("employees1.ods", engine="odf", index=False)

# print("employees.ods created successfully!")
