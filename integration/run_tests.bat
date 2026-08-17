@echo off
setlocal
cd /d %~dp0
python -m unittest discover -s tests -p "test_*.py" -v
endlocal
