@echo off
cd /d D:\Project\Android_Projects\selfy_ai
call venv310\Scripts\activate
uvicorn fastapi_app:app --reload
pause
