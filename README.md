# Bước 1: Tải mã nguồn và file mô hình
- Tải mã nguồn
```powershell
git pull https://github.com/chuonghoai/brain_tumor_app.git
cd brain_tumor_app
```
- Tải file mô hình
    - Tải folder models tại `https://drive.google.com/file/d/1GvILWQIUqR1pjP_V4Xh3yxQc1RPfSKbv/view?usp=sharing`
    - Giải nén file rar vừa tải vào project `brain_tumor_app`
    - Cấu trúc thư mục chuẩn:
        ```
        |__services/
        |__models/
        |__app.py
        |__.gitignore
        |__requirements.txt
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