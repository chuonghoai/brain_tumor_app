# Bước 1: Tải mã nguồn
```powershell
git pull https://github.com/chuonghoai/brain_tumor_app.git
cd brain_tumor_app
```

# Bước 2: Tạo môi trường ảo
- **LƯU Ý:** 
    - Nếu môi trường python đang là 3.12.5, có thể bỏ qua bước này
    - Nếu không có python 3.12.5, vui lòng cài đặt trên trang chủ https://www.python.org/
- **CÁCH TẠO VÀ KÍCH HOẠT MÔI TRƯỜNG ẢO:**
    - Mở Terminal
    - Nhập lệnh tạo môi trường ảo:
```powershell
python -m venv .venv
```

- Sau khi tạo xong, nhập lệnh sau để kích hoạt:

```powershell
.\.venv\Scripts\Activate.ps1
```
*(Lưu ý: Nếu bị lỗi ExecutionPolicy trên Windows, hãy chạy lệnh `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` trước khi kích hoạt)*

# Bước 3: Cài đặt thư viện
- Mở Terminal
- Chạy lệnh:
```powershell
pip install -r requirements.txt
```

# Bước 4: Chạy chương trình
- Chạy lệnh:
```powershell
streamlit run app.py
```
- Trang web sẽ được chạy trên trình duyệt với địa chỉ: http://localhost:8501/