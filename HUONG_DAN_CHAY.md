# Hướng Dẫn Chạy Training và Inference

## 📋 Tổng Quan

Code này được thiết kế cho **Google Cloud TPU v3-8** và **TensorFlow 1.15**. Để chạy trên Colab/Kaggle, bạn cần:
- TensorFlow 1.15 (hoặc tương thích)
- TPU runtime (Colab) hoặc GPU (Kaggle)
- Google Cloud Storage bucket (hoặc sửa code để dùng local storage)

---

## 🎯 Option 1: Inference với Model Pretrained (KHUYẾN NGHỊ)

### Bước 1: Download Pretrained Model

Models và samples có sẵn tại: https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja

**Cách download:**
```bash
# Trên Colab/Kaggle
!wget -O model.zip "https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja?dl=1"
!unzip model.zip
```

### Bước 2: Chạy Inference

#### Cách 1: Dùng `simple_eval` (đơn giản nhất)

```bash
# Ví dụ cho CIFAR-10
python3 scripts/run_cifar.py simple_eval \
  --model_dir /path/to/model/checkpoint \
  --tpu_name your-tpu-name \
  --bucket_name_prefix your-bucket-prefix \
  --mode progressive_samples \
  --load_ckpt model.ckpt-1000000 \
  --total_bs 64
```

**Các mode có sẵn:**
- `progressive_samples`: Tạo samples và lưu progressive predictions
- `bpd_train`: Tính bits-per-dimension trên training set
- `bpd_eval`: Tính bits-per-dimension trên eval set

#### Cách 2: Dùng `evaluation` (đầy đủ hơn)

```bash
# Tạo samples và tính metrics
python3 scripts/run_cifar.py evaluation \
  --model_dir /path/to/model/checkpoint \
  --tpu_name your-tpu-name \
  --bucket_name_prefix your-bucket-prefix \
  --once True \
  --dump_samples_only True \
  --total_bs 64
```

**Tham số quan trọng:**
- `--model_dir`: Đường dẫn đến thư mục chứa checkpoint
- `--tpu_name`: Tên TPU (hoặc `None` nếu chạy trên CPU/GPU)
- `--bucket_name_prefix`: Prefix của GCS bucket
- `--load_ckpt`: Tên checkpoint (ví dụ: `model.ckpt-1000000`)
- `--once`: Chỉ chạy 1 lần (không loop)
- `--dump_samples_only`: Chỉ tạo samples, không tính metrics

---

## 🚀 Option 2: Training từ đầu

### Bước 1: Setup Environment

```bash
# Cài đặt dependencies
pip3 install fire scipy pillow
pip3 install tensorflow-probability==0.8
pip3 install tensorflow-gan==0.0.0.dev0
pip3 install tensorflow-datasets==2.1.0
```

**Lưu ý:** TensorFlow 1.15 có thể không tương thích với Python mới. Cân nhắc dùng Docker hoặc virtualenv.

### Bước 2: Setup GCS Bucket

```bash
# Tạo bucket trên Google Cloud
gsutil mb gs://your-bucket-prefix-us-central1

# Upload dataset (nếu cần)
gsutil cp -r /local/dataset gs://your-bucket-prefix-us-central1/tensorflow_datasets
```

### Bước 3: Chạy Training

#### CIFAR-10:
```bash
python3 scripts/run_cifar.py train \
  --exp_name my_experiment \
  --tpu_name your-tpu-name \
  --bucket_name_prefix your-bucket-prefix \
  --model_name unet2d16b2 \
  --dataset cifar10 \
  --total_bs 128 \
  --lr 2e-4 \
  --num_diffusion_timesteps 1000 \
  --beta_start 0.0001 \
  --beta_end 0.02
```

#### CelebA-HQ:
```bash
python3 scripts/run_celebahq.py train \
  --exp_name celebahq_experiment \
  --tpu_name your-tpu-name \
  --bucket_name_prefix your-bucket-prefix \
  --total_bs 64 \
  --lr 0.00002
```

#### LSUN:
```bash
python3 scripts/run_lsun.py train \
  --exp_name lsun_church \
  --tpu_name your-tpu-name \
  --bucket_name_prefix your-bucket-prefix \
  --tfr_file 'tensorflow_datasets/lsun/church/church-r08.tfrecords' \
  --total_bs 64
```

### Bước 4: Monitor Training

Checkpoints được lưu tại: `gs://your-bucket-prefix-us-central1/logs/your_experiment_name/`

Xem logs:
```bash
tensorboard --logdir=gs://your-bucket-prefix-us-central1/logs
```

---

## 💻 Chạy trên Colab

### Setup Colab Notebook:

```python
# Cell 1: Install dependencies
!pip3 install fire scipy pillow tensorflow-probability==0.8 tensorflow-gan==0.0.0.dev0 tensorflow-datasets==2.1.0

# Cell 2: Clone repo (hoặc upload code)
!git clone https://github.com/hojonathanho/diffusion.git
%cd diffusion

# Cell 3: Authenticate GCP (nếu dùng GCS)
from google.colab import auth
auth.authenticate_user()

# Cell 4: Setup TPU
import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_name = tpu.get_master()

# Cell 5: Download pretrained model (nếu có)
!wget -O model.zip "https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja?dl=1"
!unzip model.zip

# Cell 6: Chạy inference
!python3 scripts/run_cifar.py simple_eval \
  --model_dir ./checkpoints/cifar10 \
  --tpu_name $tpu_name \
  --bucket_name_prefix your-bucket \
  --mode progressive_samples \
  --load_ckpt model.ckpt-1000000
```

**Lưu ý:** Colab có thể không hỗ trợ TensorFlow 1.15. Cân nhắc upgrade code lên TF 2.x hoặc dùng runtime cũ.

---

## 🎮 Chạy trên Kaggle

Kaggle **KHÔNG có TPU**, nên bạn cần:

1. **Sửa code để chạy trên GPU/CPU** (không dùng TPU)
2. **Thay GCS bằng local storage** (`/kaggle/working/`)
3. **Cài TensorFlow 1.15** (có thể khó khăn)

**Khuyến nghị:** Chỉ dùng Kaggle để inference với model đã train sẵn, hoặc port code sang PyTorch.

---

## 📝 Các Tham Số Quan Trọng

### Training Parameters:
- `--total_bs`: Batch size tổng (chia cho số TPU cores)
- `--lr`: Learning rate (thường 2e-4 cho CIFAR, 2e-5 cho LSUN)
- `--num_diffusion_timesteps`: Số timesteps (thường 1000)
- `--beta_start`, `--beta_end`: Beta schedule range
- `--model_mean_type`: `'eps'` (predict noise) hoặc `'xstart'` (predict x_0)
- `--loss_type`: `'mse'` hoặc `'kl'`

### Inference Parameters:
- `--mode`: `progressive_samples`, `bpd_train`, `bpd_eval`
- `--load_ckpt`: Tên checkpoint file (không có extension)
- `--total_bs`: Batch size cho sampling
- `--dump_samples_only`: Chỉ tạo samples, bỏ qua metrics

---

## 🔧 Troubleshooting

### Lỗi: "TPU not found"
- Kiểm tra TPU đã được tạo và kết nối
- Trên Colab: Runtime → Change runtime type → TPU

### Lỗi: "Bucket not found"
- Kiểm tra GCS bucket đã được tạo
- Kiểm tra quyền truy cập (authentication)

### Lỗi: "TensorFlow version mismatch"
- Code cần TF 1.15, nhưng môi trường có TF 2.x
- Giải pháp: Dùng Docker hoặc virtualenv với TF 1.15

### Lỗi: "Checkpoint not found"
- Kiểm tra đường dẫn `--model_dir` đúng
- Kiểm tra file checkpoint tồn tại: `model.ckpt-*.index`, `model.ckpt-*.data-*`

---

## 📚 Tài Liệu Tham Khảo

- Paper: https://arxiv.org/abs/2006.11239
- Website: https://hojonathanho.github.io/diffusion
- Pretrained Models: https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja

---

## ⚠️ Lưu Ý Quan Trọng

1. **Code này rất cũ** (TF 1.15, 2020), có thể không chạy được trên môi trường hiện đại
2. **Cần TPU hoặc GPU mạnh** để training
3. **GCS bucket là bắt buộc** trừ khi bạn sửa code để dùng local storage
4. **Khuyến nghị:** Nếu chỉ cần inference, download pretrained model và chạy `simple_eval`


