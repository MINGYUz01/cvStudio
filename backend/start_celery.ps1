# Celery Worker 启动脚本
# 使用 cvstudio conda 环境运行

$envPath = "D:\miniconda3\envs\cvstudio\python.exe"
$backendPath = "F:\claude_projects\cvStudio\backend"

Write-Host "正在启动 Celery Worker..." -ForegroundColor Cyan
Write-Host "Python: $envPath" -ForegroundColor Gray
Write-Host "工作目录: $backendPath" -ForegroundColor Gray
Write-Host ""

# 设置工作目录
Set-Location $backendPath

# 使用 cvstudio 环境的 Python 运行 Celery
& $envPath -m celery -A celery_app worker -l info --pool=solo
