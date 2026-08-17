@echo off
setlocal
cd /d %~dp0
python real_metrics_agent.py --host 0.0.0.0 --port 8090 %*
endlocal
