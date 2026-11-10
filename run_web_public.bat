@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo  Mo website + tao link cong khai (Cloudflare Quick Tunnel)
echo  KHONG dong cua so nay trong luc demo.
echo ============================================================
start "Website Streamlit" cmd /k python -m streamlit run app.py --server.headless true
echo Dang khoi dong web, cho 10 giay...
timeout /t 10 /nobreak >nul
echo.
echo Tim dong co duong link dang https://....trycloudflare.com ben duoi
echo va gui link do sang laptop (Zalo/Messenger/email cho chinh minh).
echo.
cloudflared tunnel --url http://localhost:8501
pause
