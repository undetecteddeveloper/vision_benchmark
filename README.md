# So sánh khả năng định vị bbox: DeepSeek V4.1 Flash vs Gemini 3.7 Flash

Bộ 3 script này chạy **cùng một** bộ ảnh, **cùng một** prompt, **cùng một**
cách chấm điểm (pycocotools COCOeval - chính là công cụ đứng sau con số
mAP@50 mà các bảng xếp hạng như Roboflow Vision Evals công bố) cho cả hai
model, để bạn có một con số mAP thật sự đối chiếu được, do chính bạn đo,
thay vì ghép hai benchmark khác nhau của hai bên công bố.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Bước 1 - Lấy dữ liệu ground truth (làm 1 lần)

1. Tải bộ annotation của COCO val2017 (~241 MB, không cần tải ảnh):
   https://images.cocodataset.org/annotations/annotations_trainval2017.zip
2. Giải nén, copy file `annotations/instances_val2017.json` vào cùng
   thư mục với các script này.
3. Chạy:

   ```bash
   python setup_data.py --n 50 --min-objects 3 --seed 42
   ```

   Script sẽ chọn ngẫu nhiên 50 ảnh có ít nhất 3 vật thể được gán nhãn
   (để bài test thực sự đo "phát hiện nhiều vật thể", không phải ảnh chỉ
   có 1 con vật), tải riêng 50 ảnh đó (không cần tải hết 5000 ảnh của
   val2017), và ghi ra `data/ground_truth.json`.

   Muốn mẫu lớn hơn cho kết quả ổn định hơn thì tăng `--n` (100-200 là hợp
   lý cho một lần test cá nhân; muốn khớp gần với quy mô Roboflow dùng
   thì cần hàng nghìn ảnh).

## Bước 2 - Gọi hai model

```bash
export DEEPSEEK_API_KEY=...   # platform.deepseek.com
python query_models.py --model deepseek

export GEMINI_API_KEY=...     # aistudio.google.com
python query_models.py --model gemini
```

Mỗi lần chạy ghi ra `data/predictions_<model>.json` (bbox dự đoán, định
dạng COCO results) và `data/failures_<model>.json` (ảnh nào bị lỗi gọi
API, hoặc model trả về nhãn/box không hợp lệ nên bị loại). File failures
đáng xem riêng: một model "né" ảnh khó bằng cách trả về ít box hơn có thể
vẫn được mAP cao giả tạo nếu chỉ nhìn con số cuối.

## Bước 3 - Chấm điểm và so sánh

```bash
python evaluate.py
```

In ra bảng mAP@[.50:.95], mAP@.50 (= "mAP@50" trong các bảng xếp hạng),
mAP@.75, AR, chia theo kích thước vật thể (small/medium/large), cho cả
hai model cạnh nhau.

## Vì sao cách này công bằng hơn so sánh 2 con số công bố sẵn

- **Cùng ảnh, cùng ground truth**: cả hai model được test trên đúng một
  bộ ảnh, nên không lệch do khác tập dữ liệu.
- **Cùng prompt, cùng schema output**: cả hai bị yêu cầu trả bbox theo
  đúng một định dạng pixel `[x_min, y_min, x_max, y_max]` do prompt quy
  định, thay vì để mỗi model dùng quy ước riêng của nó (Gemini vốn quen
  trả về `[ymin, xmin, ymax, xmax]` chuẩn hoá 0-1000) - nên chênh lệch đo
  được là do khả năng định vị thật, không phải do model "quen" hay
  "không quen" định dạng đang được hỏi.
- **Cùng công cụ chấm điểm**: pycocotools là chính công cụ chuẩn ngành
  dùng để tính mAP, nên số bạn ra được nói cùng "ngôn ngữ" với số
  Roboflow từng công bố cho Gemini 3.7 Flash (69.4% mAP@50) - dù mẫu của
  bạn nhỏ hơn nhiều nên đừng kỳ vọng khớp chính xác, chỉ nên dùng để so
  sánh *tương đối* giữa hai model trong cùng lần chạy của bạn.

## Lưu ý

- Mẫu 50 ảnh cho kết quả tham khảo nhanh, không đủ lớn để kết luận chắc
  chắn ở mức phần trăm lẻ - coi khoảng cách vài điểm mAP là nhiễu ngẫu
  nhiên, khoảng cách chục điểm mới đáng tin.
- Model ID dùng trong `query_models.py` (`deepseek-flash`,
  `gemini-3.7-flash`) là tên gọi tại thời điểm viết script này; nếu nhà
  cung cấp đổi tên, chỉ cần sửa 2 chỗ đó.
- `temperature=0` được đặt cho cả hai để kết quả tái lập được giữa các
  lần chạy.
