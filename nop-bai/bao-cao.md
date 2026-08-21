# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Lê Mai Việt Hoàng |
| MSSV | 2A202601230 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/hoanglmv/Track2-DAY21-2A202601230-LeMaiVietHoang |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.10 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.10 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.

**Lý do:** Bộ siêu tham số này đạt $F_1$-score ấn tượng ($0.7109$), vượt xa ngưỡng chất lượng $0.65$ và đạt Accuracy cao nhất ($0.8780$). So với lần 3 ($n\_estimators=200, max\_depth=5$ đạt $F_1=0.7149$), lần 1 có độ phức tạp thấp hơn nhiều (độ sâu cây 3 thay vì 5), thời gian huấn luyện nhanh và giảm thiểu rủi ro quá khớp (overfitting). Ta cũng nhận thấy sự đánh đổi rõ rệt: ở lần 2 khi giảm `learning_rate` xuống $0.05$ và chỉ dùng $50$ cây nông ($depth=2$), mô hình bị underfit nghiêm trọng, khiến $F_1$-score tụt xuống $0.6051$ (dưới ngưỡng triển khai), trong khi Accuracy vẫn ở mức $0.8460$.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng nghiêm trọng: chỉ có $24.8\%$ số mẫu thuộc lớp dương (thu nhập $>50K$), trong khi $75.2\%$ thuộc lớp âm ($\le 50K$). Do đó, một mô hình "vô dụng" luôn đưa ra dự đoán thu nhập thấp cho mọi trường hợp vẫn dễ dàng đạt được Accuracy rất cao là $0.752$ ($75.2\%$), nhưng hoàn toàn không học được gì và có $F_1 = 0.0$. 

$F_1$-score của lớp dương là trung bình điều hòa giữa Precision và Recall, phản ánh chính xác khả năng nhận diện đúng nhóm đối tượng thu nhập cao mà Accuracy không thể đo lường được. Trong bài toán này, ta không sử dụng `average="weighted"` hay `average="macro"` vì các trọng số từ lớp đa số sẽ kéo chỉ số lên cao giả tạo, làm mất đi ý nghĩa giám sát nghiêm ngặt của Quality Gate.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi `ModuleNotFoundError: No module named 'pkg_resources'` khi import MLflow | Python 3.12 và setuptools mới (>=70) đã loại bỏ module `pkg_resources` | Khóa phiên bản `setuptools<70` (cụ thể 69.5.1) trong `requirements.txt` |
| Lỗi `AccessDenied` khi tạo S3 Bucket qua AWS CLI | IAM User `hoanglmv` chưa được gán chính sách phân quyền | Gắn chính sách `AdministratorAccess` cho user trên AWS IAM Console |
| Lỗi `AttributeError` khi unpickle model GradientBoosting trên EC2 | Phiên bản `scikit-learn` trên EC2 không đồng nhất với môi trường huấn luyện | Cài đặt chính xác `scikit-learn==1.4.2` khớp với `requirements.txt` trên EC2 |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1` - 22.361 mẫu) | 0.7109 | 0.8780 |
| Bước 3 (thêm `train_batch2` - 44.722 mẫu) | 0.7118 | 0.8790 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu từ `train_batch2`, chỉ số $F_1$-score và Accuracy duy trì độ ổn định cao và tăng nhẹ ($F_1$ đạt $0.7118$, Accuracy đạt $0.8790$). Do cả hai batch dữ liệu đều được phân chia ngẫu nhiên từ cùng một phân phối dân số, mô hình cơ bản đã nắm bắt được quy luật cốt lõi từ batch 1. Ý nghĩa quan trọng nhất ở Bước 3 là đã kiểm chứng thành công pipeline Continuous Training: toàn bộ quá trình từ nạp dữ liệu DVC, kiểm thử, huấn luyện lại cho đến triển khai tự động lên EC2 đều diễn ra liền mạch mà không cần bất kỳ can thiệp thủ công nào.
