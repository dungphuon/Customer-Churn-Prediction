# Churn Prediction Project 
## 📝 Giới thiệu
Dự án này tập trung vào việc xây dựng mô hình học máy để dự đoán tỉ lệ khách hàng rời bỏ (Churn Prediction). Đây là một bài toán thực tế giúp doanh nghiệp xác định những khách hàng có khả năng ngừng sử dụng dịch vụ, từ đó đưa ra các chính sách giữ chân kịp thời.

Dự án bao gồm toàn bộ quy trình từ xử lý dữ liệu thô, phân tích đặc trưng (EDA), huấn luyện mô hình cho đến triển khai ứng dụng minh họa.

## 📂 Cấu trúc thư mục
Dựa trên cấu trúc dự án:
*   `churn_app/`: Chứa mã nguồn của ứng dụng giao diện (Streamlit) để dự đoán và hiển thị dashboard.
*   `models/`: Thư mục lưu trữ các mô hình đã được huấn luyện thành công.
*   `data_cleaning.ipynb`: Notebook thực hiện các bước làm sạch và tiền xử lý dữ liệu ban đầu.
*   `churn_labeling.ipynb`: Notebook xử lý việc gán nhãn dữ liệu phù hợp với bài toán.
*   `model_experiment.ipynb`: Notebook thực hiện các thử nghiệm, so sánh và đánh giá hiệu năng giữa các thuật toán học máy.
*   `customer_churn.csv`: Tập dữ liệu gốc được sử dụng cho dự án.
*   `scaler.pkl`: File lưu trữ bộ chuẩn hóa dữ liệu (StandardScaler/MinMaxScaler) để đồng bộ dữ liệu khi đưa vào model.

## 🛠 Công nghệ sử dụng
*   **Ngôn ngữ:** Python
*   **Thư viện chính:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
*   **Môi trường làm việc:** Jupyter Notebook, Visual Studio Code.
*   **Triển khai:** Streamlit (Ứng dụng web dự đoán).

🚀 Hướng dẫn cài đặt và chạy
Khởi tạo môi trường ảo (Virtual Environment):

PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Cài đặt các thư viện cần thiết:

PowerShell
pip install -r requirements.txt
Chạy ứng dụng web (trong thư mục churn_app):

PowerShell
streamlit run churn_app/app.py
