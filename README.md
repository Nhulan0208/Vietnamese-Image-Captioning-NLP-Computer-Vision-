# Vietnamese-Image-Captioning-NLP-Computer-Vision-

> Sinh mô tả ảnh tự động bằng **tiếng Việt** sử dụng **ViT-B/16 + Transformer Decoder** trên dataset UIT-ViIC

---

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc model](#kiến-trúc-model)
- [Dataset](#dataset--uit-viic)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Kỹ thuật huấn luyện](#kỹ-thuật-huấn-luyện)
- [Đánh giá](#đánh-giá)
- [Flask API & Deployment](#flask-api--deployment)
- [Kết quả](#kết-quả)
- [Hướng phát triển](#hướng-phát-triển)

---

## Tổng quan

Dự án xây dựng pipeline end-to-end để sinh **caption tiếng Việt** tự động cho ảnh thể thao, gồm 2 giai đoạn:

| Giai đoạn | Notebook | Nội dung |
|-----------|----------|----------|
| 1 | `DataTrain.ipynb` | EDA → Tiền xử lý → Huấn luyện → Xuất model |
| 2 | `EvalApp.ipynb` | Load model → Đánh giá BLEU/METEOR → Flask API + ngrok |

**Thách thức chính:** Dataset UIT-ViIC bị mất cân bằng nghiêm trọng (Tennis chiếm ~81.5%), đòi hỏi các kỹ thuật đặc biệt để xử lý.

---

## Kiến trúc model

```
Input Image (224×224)
        │
        ▼
┌───────────────────────────────┐
│  ViT-B/16 Encoder (Frozen)    │
│  16×16 patches → 196 tokens   │
│  768-dim → Linear → 512-dim   │
└──────────────┬────────────────┘
               │ Memory (B, 196, 512)
               ▼
┌───────────────────────────────┐
│  Transformer Decoder          │
│  3 layers | 8 heads           │
│  d_model=512 | FFN=2048       │
│  dropout=0.35 | norm_first    │
└──────────────┬────────────────┘
               │
               ▼
     Linear(vocab_size)
               │
        ┌──────┴──────┐
        ▼             ▼
   Greedy Search   Beam Search
                  (size=3, α=0.7)
               │
               ▼
    Caption tiếng Việt 🇻🇳
```

### Chi tiết các thành phần

**ViT-B/16 Encoder**
- Pretrained trên ImageNet (frozen epoch 1–7, unfreeze 4 layer cuối từ epoch 8+)
- Input: ảnh 224×224 → 196 patches 16×16
- Output: `(B, 196, 512)` image features

**Transformer Decoder**
- Token Embedding + Positional Encoding (max_len=42)
- Masked Self-Attention → Cross-Attention(Memory) → FFN
- Output: phân phối xác suất trên toàn bộ vocabulary

**Vocabulary**
- Token đặc biệt: `<PAD>=0`, `<SOS>=1`, `<EOS>=2`, `<UNK>=3`
- Lọc từ có tần suất ≥ 2
- Giữ nguyên dấu tiếng Việt (Unicode)

---

## Dataset — UIT-ViIC

Bộ dữ liệu ảnh thể thao có chú thích tiếng Việt, dựa trên COCO.

| Tập | Ảnh | Captions |
|-----|-----|---------|
| Train | Từ `uitviic_captions_train2017.json` | Nhiều caption/ảnh |
| Val | Từ `uitviic_captions_val2017.json` | — |
| Test | Từ `uitviic_captions_test2017.json` | — |

**Đặc điểm:**
- Mỗi ảnh có nhiều caption tiếng Việt
- Độ dài trung bình: ~8–12 từ/caption
- **⚠️ Mất cân bằng:** Tennis ~81.5%, các môn còn lại thiểu số
- Ảnh được tải tự động từ COCO/Flickr URL

**Files cần chuẩn bị:**
```
uitviic_captions_train2017.json
uitviic_captions_val2017.json
uitviic_captions_test2017.json
uitviic_images/          ← tải tự động khi chạy notebook
```

---

## Cấu trúc dự án

```
Vietnamese-Image-Captioning/
│
├── DataTrain.ipynb          # Notebook 1: EDA + Training (Google Colab)
├── EvalApp.ipynb            # Notebook 2: Evaluation + Flask API (Colab/Local)
├── README.md
│
├── uitviic_captions_train2017.json
├── uitviic_captions_val2017.json
├── uitviic_captions_test2017.json
│
├── uitviic_images/          # Ảnh tải về (tự tạo khi chạy)
│
└── outputs/                 # Model outputs (tự tạo khi train)
    ├── checkpoint.pth       # Checkpoint sau mỗi epoch
    ├── best_model.pth       # Best validation loss
    └── vicaptioning_full.pth  # Model cuối dùng cho Notebook 2
```

---

## Cài đặt

### Yêu cầu

```bash
pip install torch torchvision
pip install nltk matplotlib seaborn wordcloud tqdm Pillow
pip install flask pyngrok
```

Tải dữ liệu NLTK (chạy một lần):
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Môi trường

| Notebook | Môi trường khuyến nghị | Lý do |
|----------|----------------------|-------|
| `DataTrain.ipynb` | Google Colab (GPU) | Training cần VRAM cao |
| `EvalApp.ipynb` | Google Colab hoặc PyCharm local | Load model + inference |

---

## Hướng dẫn sử dụng

### Bước 1 — Huấn luyện (`DataTrain.ipynb`)

1. Upload 3 file JSON lên Colab (hoặc mount Google Drive)
2. Chạy từng cell theo thứ tự:
   - **Cell 0:** Cài đặt thư viện
   - **Cell 1a:** Upload JSON + load dữ liệu
   - **Cell 1b:** Tải ảnh song song (16 threads, tự động skip ảnh đã có)
   - **Cell 2:** EDA — phân tích phân bố, WordCloud, ảnh mẫu
   - **Cell 3:** Tiền xử lý + cân bằng class (WeightedSampler)
   - **Cell 4:** Xây dựng ViCaptioningModel (ViT + Transformer Decoder)
   - **Cell 5:** Huấn luyện với Early Stopping + Checkpoint
   - **Cell 6:** Xuất `vicaptioning_full.pth`

3. **Output:** `vicaptioning_full.pth` — dùng cho Notebook 2

### Bước 2 — Đánh giá & Deploy (`EvalApp.ipynb`)

**Yêu cầu:** `vicaptioning_full.pth` từ Bước 1

1. Mount Google Drive (nếu chạy Colab)
2. Load model từ file `.pth`
3. Chạy đánh giá BLEU / METEOR (Greedy + Beam Search)
4. Visualize kết quả định tính
5. Khởi động Flask API + ngrok tunnel

```python
# Sinh caption cho ảnh mới
caption = model.generate_beam(image_tensor, vocab)
print(caption)
```

---

## Kỹ thuật huấn luyện

### Xử lý mất cân bằng dữ liệu

| Kỹ thuật | Chi tiết |
|----------|----------|
| **WeightedRandomSampler** | sqrt-smoothing trên class frequency, cân bằng 11 nhóm thể thao |
| **Focal Loss** | γ=2.0, giảm trọng số mẫu dễ (tennis) |
| **Label Smoothing** | ε=0.05, tránh overconfidence |
| **Word Weight** | Từ quan trọng (tennis, bóng đá…) nhân ×2.5 |

### Tối ưu huấn luyện

| Kỹ thuật | Chi tiết |
|----------|----------|
| **Optimizer** | AdamW |
| **Scheduler** | CosineAnnealingWarmRestarts (T₀=10, η_min=1e-7) |
| **Scheduled Sampling** | Teacher Forcing Ratio: 1.0 → 0.5 (giảm 0.033/epoch) |
| **Token Dropout** | 10%, tránh overfit |
| **Gradient Clipping** | norm=1.0, ổn định training |
| **AMP** | Automatic Mixed Precision, tăng tốc GPU |
| **Early Stopping** | Theo val_loss, tự động lưu best checkpoint |

### Progressive Unfreeze ViT

```
Epoch 1–7:  ViT backbone FROZEN  → chỉ train decoder + projection layer
Epoch 8+:   Unfreeze 4 layers cuối ViT với LR rất nhỏ (×0.03)
            → Fine-tune toàn bộ model tránh catastrophic forgetting
```

---

## Đánh giá

### Chỉ số đánh giá

| Metric | Mô tả |
|--------|-------|
| **BLEU-1/2/3/4** | Độ chính xác n-gram, tính Corpus BLEU với Smoothing Function |
| **METEOR** | Kết hợp Precision & Recall, hỗ trợ Stemming & Synonyms (wordnet) |

### Greedy vs Beam Search

| | Greedy Search | Beam Search (k=3) |
|-|--------------|-------------------|
| **Tốc độ** | Nhanh — O(n) | Chậm hơn |
| **BLEU-4** | Thấp hơn | **Cao hơn** ✅ |
| **METEOR** | Thấp hơn | **Cao hơn** ✅ |
| **Đặc điểm** | Luôn chọn token max prob | Duy trì top-3 beam, length-normalize cuối |

**Beam Search được cải tiến:**
- Length normalization: `score / len^α` (α=0.7) → tránh ưu tiên câu ngắn
- EOS handling đúng: beam kết thúc vào `done`, không mở rộng tiếp
- Sort bằng raw score khi pruning, chỉ normalize khi chọn best cuối

---

## Flask API & Deployment

Model được wrap thành **REST API** và expose ra internet qua **ngrok tunnel**.

```
POST /predict
Content-Type: multipart/form-data

Body: { image: <file> }
Response: { "caption": "Một cầu thủ đang đánh bóng tennis." }
```

**Tính năng API:**
- ⚡ Device auto-detect: GPU (CUDA) nếu có, fallback CPU
- 🔄 Model load một lần khi khởi động, inference nhanh
- 📊 Hỗ trợ cả Greedy và Beam Search qua query parameter
- 🛡️ Error handling: ảnh không hợp lệ, timeout, exception an toàn
- 🌐 ngrok URL public để demo trực tiếp không cần server riêng

---

## Kết quả

### Tóm tắt kiến trúc

| Thành phần | Cấu hình |
|------------|---------|
| **Encoder** | ViT-B/16, ImageNet pretrained, 196 patches × 512 dim |
| **Decoder** | 3-layer Transformer, nhead=8, d=512, dropout=0.35 |
| **Tìm kiếm** | Beam Search, size=3, α=0.7, length normalized |
| **Loss** | Focal Loss (γ=2.0) + Label Smoothing (ε=0.05) |

### Bài học rút ra

- ✅ ViT-B/16 + Transformer Decoder là pipeline hiệu quả cho Image Captioning tiếng Việt
- ✅ Focal Loss + SmartSampler giải quyết được mất cân bằng nghiêm trọng (Tennis 81.5%)
- ✅ Beam Search với length normalization cho kết quả rõ ràng hơn Greedy
- ✅ Progressive Unfreeze ViT giúp fine-tune hiệu quả, tránh catastrophic forgetting
- ✅ Flask + ngrok cho phép demo nhanh mà không cần server riêng

---

## Hướng phát triển

- 🔹 Mở rộng dataset với nhiều môn thể thao hơn
- 🔹 Thử nghiệm ViT-L/16 hoặc CLIP encoder
- 🔹 Cross-attention visualization (giải thích vùng ảnh model chú ý)
- 🔹 Tích hợp PhoBERT tokenizer cho tiếng Việt tốt hơn
- 🔹 Streamlit / Gradio UI thân thiện hơn Flask

---

## Tác giả

**Nguyễn Thị Như Lan**  
Môn học: NLP × Computer Vision  
Dataset: [UIT-ViIC](https://github.com/uitnlp/UIT-ViIC)
