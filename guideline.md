# Sổ Tay Hướng Dẫn Chạy Lệnh Toàn Diện - Day 21: MLOps CI/CD (AWS)

> **Ghi chú**: Toàn bộ mã nguồn trong `src/train.py`, `tests/test_train.py`, `src/serve.py` và `.github/workflows/cicd.yml` **đã được cấu hình hoàn chỉnh để hỗ trợ AWS (S3 + EC2)**. Bạn chỉ cần làm theo các hướng dẫn dưới đây.

---

## Mục Lục
1. [Khởi Tạo Môi Trường & Dữ Liệu Ban Đầu](#1-khởi-tạo-môi-trường--dữ-liệu-ban-đầu)
2. [Bước 1: Thực Nghiệm Cục Bộ & Theo Dõi MLflow](#2-bước-1-thực-nghiệm-cục-bộ--theo-dõi-mlflow)
3. [Bước 2: Thiết Lập AWS (S3, EC2), DVC & Kích Hoạt CI/CD Pipeline](#3-bước-2-thiết-lập-aws-s3-ec2-dvc--kích-hoạt-cicd-pipeline)
   - [3.1 Lấy AWS Access Key & Cài Đặt AWS CLI](#31-lấy-aws-access-key--cài-đặt-aws-cli)
   - [3.2 Hướng Dẫn Tạo S3 Bucket (Giao Diện Web & Dòng Lệnh)](#32-hướng-dẫn-tạo-s3-bucket-giao-diện-web--dòng-lệnh)
   - [3.3 Cấu Hình DVC Đẩy Dữ Liệu Lên S3](#33-cấu-hình-dvc-đẩy-dữ-liệu-lên-s3)
   - [3.4 Tạo EC2 Instance & Cấu Hình FastAPI Service](#34-tạo-ec2-instance--cấu-hình-fastapi-service)
   - [3.5 Khai Báo 5 GitHub Secrets Cho AWS](#35-khai-báo-5-github-secrets-cho-aws)
   - [3.6 Commit & Push Kích Hoạt CI/CD Pipeline Bước 2](#36-commit--push-kích-hoạt-cicd-pipeline-bước-2)
4. [Bước 3: Thêm Dữ Liệu Mới & Continuous Training](#4-bước-3-thêm-dữ-liệu-mới--continuous-training)
5. [Tổng Hợp 5 Ảnh Chụp Màn Hình & Checklist Nộp Bài](#5-tổng-hợp-5-ảnh-chụp-màn-hình--checklist-nộp-bài)

---

## 1. Khởi Tạo Môi Trường & Dữ Liệu Ban Đầu

Mở Terminal tại thư mục gốc repository (`/home/myvh07/Vinlab/Track2-DAY21-2A202601230-LeMaiVietHoang`):

```bash
# 1. Kích hoạt môi trường ảo
source .venv/bin/activate

# 2. Cài đặt các thư viện cần thiết (bao gồm boto3 và dvc[s3])
pip install -r requirements.txt

# 3. Tải và chia tập dữ liệu ban đầu
python prepare_data.py
```

---

## 2. Bước 1: Thực Nghiệm Cục Bộ & Theo Dõi MLflow

### 2.1 Cấu hình biến môi trường MLflow
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
```

### 2.2 Chạy 3 lần thí nghiệm

* **Lần 1: Chạy với tham số mặc định (100 cây, lr 0.1, depth 3)**
  ```bash
  python src/train.py
  ```

* **Lần 2: Đổi tham số thành (50 cây, lr 0.05, depth 2) và chạy**
  ```bash
  cat << 'EOF' > params.yaml
  n_estimators: 50
  learning_rate: 0.05
  max_depth: 2
  EOF

  python src/train.py
  ```

* **Lần 3: Đổi tham số thành (200 cây, lr 0.1, depth 5) và chạy**
  ```bash
  cat << 'EOF' > params.yaml
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 5
  EOF

  python src/train.py
  ```

### 2.3 Khởi chạy MLflow UI & Thu thập ảnh chụp 01
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
1. Truy cập: `http://localhost:5000` trên trình duyệt.
2. Bấm vào nút **Columns** (góc phải bảng), tick chọn các cột: `n_estimators`, `learning_rate`, `max_depth`, `f1_score`, `accuracy`.
3. Sắp xếp theo `f1_score` giảm dần.
4. **📸 CHỤP ẢNH MÀN HÌNH 01**:
   * Lưu ảnh vào file: `nop-bai/anh-chup-man-hinh/01-mlflow-ui.png`
   * Yêu cầu: Thấy URL trình duyệt `localhost:5000`, thấy tối thiểu 3 runs kèm đủ các cột params và metrics.
5. Đặt lại `params.yaml` về bộ tham số có $F_1$ cao nhất (đảm bảo $F_1 \ge 0.65$):
   ```bash
   cat << 'EOF' > params.yaml
   n_estimators: 100
   learning_rate: 0.1
   max_depth: 3
   EOF
   ```
6. Điền thông số vào **Mục 1 của file `nop-bai/bao-cao.md`**.

---

## 3. Bước 2: Thiết Lập AWS (S3, EC2), DVC & Kích Hoạt CI/CD Pipeline

### 3.1 Lấy AWS Access Key & Cài Đặt AWS CLI

#### A. Lấy Access Key trên AWS Console (nếu chưa có)
1. Đăng nhập vào [https://console.aws.amazon.com](https://console.aws.amazon.com).
2. Tìm kiếm dịch vụ **IAM** $\to$ Chọn **Users** $\to$ Chọn user của bạn.
3. Chuyển sang tab **Security credentials** $\to$ Tìm mục **Access keys** $\to$ Bấm **Create access key**.
4. Chọn **Command Line Interface (CLI)** $\to$ Tick đồng ý $\to$ Bấm **Next** $\to$ **Create access key**.
5. Lưu lại `Access key ID` và `Secret access key`.

#### B. Cài đặt và cấu hình AWS CLI
```bash
# 1. Cài đặt AWS CLI v2
sudo apt update && sudo apt install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install --update
rm -rf aws awscliv2.zip

# 2. Đăng nhập AWS CLI (nhập Access Key & Secret Key vừa lấy)
aws configure
```

---

### 3.2 Hướng Dẫn Tạo S3 Bucket (Giao Diện Web & Dòng Lệnh)

#### Cách 1: Tạo trên giao diện Web (AWS Console) - Dễ nhất
1. Truy cập [https://console.aws.amazon.com/s3/](https://console.aws.amazon.com/s3/).
2. Bấm nút màu cam **"Create bucket"**.
3. **Bucket name**: Đặt tên **duy nhất toàn cầu** (chỉ gồm chữ thường, số, gạch ngang), ví dụ: `income-mlops-hoanglmv-2026`.
4. **AWS Region**: Chọn vùng (ví dụ: `US East (N. Virginia) us-east-1`).
5. **Block Public Access**: Giữ nguyên mặc định (tick chọn Block all public access).
6. Cuộn xuống cuối trang và bấm **"Create bucket"**.

#### Cách 2: Tạo bằng dòng lệnh Terminal
```bash
export BUCKET="income-mlops-hoanglmv-2026"   # Thay bằng tên bucket duy nhất của bạn
export AWS_REGION="us-east-1"

# Lệnh tạo Bucket
aws s3 mb s3://$BUCKET --region $AWS_REGION

# Kiểm tra bucket đã tạo thành công
aws s3 ls
```

---

### 3.3 Cấu Hình DVC Đẩy Dữ Liệu Lên S3

```bash
# 1. Khởi tạo DVC và trỏ remote đến S3 Bucket
dvc init
dvc remote add -d labstore s3://$BUCKET/dvc

# 2. Thêm 3 file dữ liệu vào DVC tracking
dvc add data/train_batch1.csv
dvc add data/holdout.csv
dvc add data/train_batch2.csv

# 3. Đẩy dữ liệu CSV lên S3 Bucket
dvc push
```

---

### 3.4 Tạo EC2 Instance & Cấu Hình FastAPI Service

#### A. Khởi tạo EC2 Instance
```bash
# 1. Tạo SSH Key Pair trên AWS
aws ec2 create-key-pair --key-name income-ec2-key --query 'KeyMaterial' --output text > ~/.ssh/income-ec2-key.pem
chmod 400 ~/.ssh/income-ec2-key.pem

# 2. Tạo Security Group mở port 22 (SSH) và port 8080 (FastAPI)
export SG_ID=$(aws ec2 create-security-group --group-name income-api-sg --description "Security group for Income API" --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8080 --cidr 0.0.0.0/0

# 3. Lấy AMI ID Ubuntu 22.04 LTS mới nhất
export AMI_ID=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id --query "Parameter.Value" --output text)

# 4. Khởi chạy EC2 Instance
export INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.micro \
  --key-name income-ec2-key \
  --security-group-ids $SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=income-api}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Đang đợi EC2 khởi động..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# 5. Lấy địa chỉ IP công khai của EC2
export VM_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo ">>> IP công khai của EC2 là: $VM_IP"
```

#### B. Cấu hình FastAPI & Systemd Service Trên EC2
```bash
# 1. SSH vào EC2 cài đặt môi trường
ssh -i ~/.ssh/income-ec2-key.pem -o StrictHostKeyChecking=no ubuntu@$VM_IP \
  "sudo apt update && sudo apt install -y python3-pip && pip3 install fastapi uvicorn scikit-learn joblib boto3 && mkdir -p ~/models ~/src ~/.aws"

# 2. Copy AWS credentials và script serve.py lên EC2
scp -i ~/.ssh/income-ec2-key.pem ~/.aws/credentials ubuntu@$VM_IP:~/.aws/credentials
scp -i ~/.ssh/income-ec2-key.pem ~/.aws/config ubuntu@$VM_IP:~/.aws/config
scp -i ~/.ssh/income-ec2-key.pem src/serve.py ubuntu@$VM_IP:~/src/serve.py

# 3. Tạo systemd service income-api trên EC2
ssh -i ~/.ssh/income-ec2-key.pem ubuntu@$VM_IP << EOF
sudo tee /etc/systemd/system/income-api.service > /dev/null <<SERVICE_EOF
[Unit]
Description=Income Model Inference Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="ARTIFACT_BUCKET=$BUCKET"
Environment="AWS_DEFAULT_REGION=$AWS_REGION"
ExecStart=/usr/bin/python3 /home/ubuntu/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE_EOF

sudo systemctl daemon-reload
sudo systemctl enable income-api
EOF

# 4. Tạo SSH Key riêng cho GitHub Actions deploy
ssh-keygen -t ed25519 -f ~/.ssh/income_deploy -N "" -C "github-actions-deploy"
cat ~/.ssh/income_deploy.pub | ssh -i ~/.ssh/income-ec2-key.pem ubuntu@$VM_IP \
  "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

---

### 3.5 Khai Báo 5 GitHub Secrets Cho AWS

Vào GitHub Repository của bạn $\to$ **Settings** $\to$ **Secrets and variables** $\to$ **Actions** $\to$ Bấm **New repository secret**:

| Tên Secret | Giá trị cần điền cho AWS |
|---|---|
| `STORAGE_CREDENTIALS` | Dán chuỗi JSON chứa AWS key: `{"aws_access_key_id":"<YOUR_ACCESS_KEY>","aws_secret_access_key":"<YOUR_SECRET_KEY>","aws_default_region":"us-east-1"}` |
| `ARTIFACT_BUCKET` | Tên S3 Bucket của bạn (ví dụ: `income-mlops-hoanglmv-2026`) |
| `SERVER_HOST` | Địa chỉ IP công khai của EC2 (`echo $VM_IP`) |
| `SERVER_USER` | `ubuntu` |
| `SERVER_SSH_KEY` | Toàn bộ nội dung private key trong file `~/.ssh/income_deploy` (bắt đầu bằng `-----BEGIN OPENSSH PRIVATE KEY-----` đến hết) |

---

### 3.6 Commit & Push Kích Hoạt CI/CD Pipeline Bước 2
```bash
git add .
git commit -m "feat: complete step 2 CI/CD pipeline on AWS"
git push origin main
```

#### 📸 Thu thập các ảnh chụp màn hình cho Bước 2:
1. **📸 CHỤP ẢNH MÀN HÌNH 02 (`02-actions-buoc-2.png`)**:
   * Vào tab **Actions** trên GitHub repo.
   * Chụp ảnh toàn bộ giao diện thấy rõ 4 jobs xanh: `Unit Test`, `Train`, `Quality Gate`, `Release` cùng với Commit message.
2. **Khởi động service trên EC2 (nếu chưa chạy)**:
   ```bash
   ssh -i ~/.ssh/income-ec2-key.pem ubuntu@$VM_IP "sudo systemctl restart income-api"
   ```
3. **📸 CHỤP ẢNH MÀN HÌNH 04 (`04-curl-api.png`)**:
   * Mở terminal trên máy cá nhân và chạy 2 lệnh curl tới IP của EC2:
     ```bash
     # Kiểm tra healthcheck
     curl http://$VM_IP:8080/healthz

     # Kiểm tra dự đoán
     curl -X POST http://$VM_IP:8080/score \
       -H "Content-Type: application/json" \
       -d '{"features": [39, 5, 13, 4, 0, 1, 1, 2174, 0, 40]}'
     ```
   * Chụp ảnh terminal hiển thị 2 lệnh curl và kết quả trả về `{"prediction": ..., "label": ...}`.
4. **📸 CHỤP ẢNH MÀN HÌNH 05 (`05-cloud-storage.png`)**:
   * Mở trình duyệt vào AWS Management Console $\to$ **Amazon S3** $\to$ Chọn Bucket của bạn.
   * Chụp màn hình thấy rõ bucket chứa thư mục `dvc/` và file `artifacts/current/model.joblib`.

---

## 4. Bước 3: Thêm Dữ Liệu Mới & Continuous Training

Mục tiêu: Bổ sung 22,361 mẫu dữ liệu mới, đẩy lên S3 qua DVC và kích hoạt tự động pipeline chỉ bằng một lệnh `git push`.

```bash
# 1. Ghép batch 2 vào tập dữ liệu huấn luyện (tăng lên 44,722 mẫu)
python append_batch.py

# 2. Báo DVC cập nhật dữ liệu mới
dvc add data/train_batch1.csv

# 3. ĐẨY DỮ LIỆU LÊN AWS S3 TRƯỚC (Bắt buộc trước khi git push)
dvc push

# 4. Commit file con trỏ .dvc và push lên GitHub (Kích hoạt GitHub Actions)
git add data/train_batch1.csv.dvc
git commit -m "data: bổ sung 22361 mẫu dữ liệu mới (train_batch2)"
git push origin main
```

#### 📸 Thu thập ảnh chụp màn hình cho Bước 3:
1. **📸 CHỤP ẢNH MÀN HÌNH 03 (`03-actions-buoc-3.png`)**:
   * Vào tab **Actions** trên GitHub.
   * Chụp màn hình run mới nhất thấy rõ commit message `data: bổ sung 22361 mẫu dữ liệu mới (train_batch2)` và 4 jobs hoàn thành màu xanh.
2. Tải file `report.json` từ Artifacts của 2 lần chạy Bước 2 và Bước 3 về để đối chiếu $F_1$-score.
3. Điền bảng so sánh và giải thích vào **Mục 2 & 3 của `nop-bai/bao-cao.md`**.

---

## 5. Tổng Hợp 5 Ảnh Chụp Màn Hình & Checklist Nộp Bài

### 5.1 Danh sách 5 file ảnh bắt buộc
Lưu 5 ảnh vào thư mục `nop-bai/anh-chup-man-hinh/`:
- [x] `01-mlflow-ui.png`: Giao diện MLflow UI với 3 runs so sánh $F_1$ vs Accuracy.
- [x] `02-actions-buoc-2.png`: GitHub Actions Bước 2 với 4 jobs hoàn thành màu xanh.
- [x] `03-actions-buoc-3.png`: GitHub Actions Bước 3 được kích hoạt bởi commit dữ liệu.
- [x] `04-curl-api.png`: Terminal chạy curl gọi `/healthz` và `/score` đến IP EC2.
- [x] `05-cloud-storage.png`: AWS S3 Console hiển thị `dvc/` và `artifacts/current/model.joblib`.

### 5.2 Hoàn thiện báo cáo & Nộp bài
1. Mở file `nop-bai/bao-cao.md`, điền các nội dung phân tích, sau đó xóa các khối chú thích hướng dẫn (đảm bảo báo cáo gọn gàng, không quá 1 trang A4).
2. Commit và push toàn bộ ảnh và báo cáo lên GitHub:
   ```bash
   git add nop-bai/
   git commit -m "docs: hoàn thiện báo cáo và ảnh chụp màn hình nộp bài"
   git push origin main
   ```
3. Đảm bảo GitHub Repository ở chế độ **Public**.
4. Nộp link Repository lên hệ thống **https://codelabs.vlearn.dev**.
