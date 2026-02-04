#!/usr/bin/env python3
"""
Test script to verify parallel processing sends files_progress data
"""
import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Find all progress keys
keys = r.keys('progress:*')
print(f"Found {len(keys)} progress tasks in Redis\n")

for key in keys[:5]:  # Show first 5
    data = r.get(key)
    if data:
        progress = json.loads(data)
        task_id = progress.get('task_id', 'unknown')
        has_files_progress = 'files_progress' in progress
        files_count = len(progress.get('files_progress', [])) if has_files_progress else 0
        
        print(f"Task: {task_id}")
        print(f"  - Has files_progress: {has_files_progress}")
        print(f"  - Files tracked: {files_count}")
        print(f"  - Overall progress: {progress.get('progress_percentage', 0)}%")
        
        if has_files_progress and files_count > 0:
            print(f"  - Files progress detail:")
            for fp in progress['files_progress'][:3]:  # Show first 3 files
                if fp:
                    print(f"    • {fp['filename']}: {fp['progress']}% ({fp['status']})")
        print()
