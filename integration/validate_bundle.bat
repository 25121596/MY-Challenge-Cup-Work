@echo off
cd /d "%~dp0"
python validate_bundle_runtime.py %*
pause
