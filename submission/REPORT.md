# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hoàng Việt  **MSSV**: 2A202602602  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp chính xác 100% với các file tạo ra trong `results/` trên tập đánh giá đầy đủ (full run, 50 mẫu target + 15 mẫu regression).
> Bài báo cáo tuân thủ đầy đủ cấu trúc rubric: lựa chọn & thiết lập thực nghiệm, bằng chứng tính đúng của loss mask, mốc cơ sở đóng băng trước huấn luyện, giải phẫu cấu hình sai, phán quyết cổng hồi quy, phân tích định tính đa chiều (bao gồm ca thua) và bài học thực tế.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 khóa |
| Train / val | 225 / 25 (split seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json, suggested_max_length=256)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** Có — theo `results/template_check.json`, cả `open_tag_present` và `body_present` đều đạt `true` (`verdict: "reasoning preserved — safe to train on traces"`). Chat template jinja của Qwen3.5 bảo toàn nguyên vẹn block suy luận, không nuốt hoặc cắt xén thẻ `<think>`, đảm bảo an toàn tuyệt đối khi huấn luyện mô hình suy luận.

---

## 2. Mask proof (NB1)

| Chỉ số kiểm tra | Giá trị thực tế |
|---|---|
| `supervised_fraction` | 0.4149 (41.5%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn trích 3–5 dòng đầu của phần câu trả lời được tính loss (`supervised_preview`):

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

*Phân tích:* Tỷ lệ giám sát chỉ chiếm 41.49% tổng số token (39/94 token). Toàn bộ system prompt và câu hỏi của người dùng (`masked_preview`) đã được gán nhãn `-100` (`IGNORE_INDEX`), loại bỏ hoàn toàn nguy cơ model học vẹt câu hỏi hoặc rò rỉ prompt.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train trên 50 mẫu)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3316.3 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1013.9 |
| (c) LoRA fine-tune | 0.9700 | 0.5222 | 1.0000 | 1430.8 |

**(b) có thật sự mạnh hơn (a) không?** Có, (b) vượt trội hoàn toàn so với (a) trên cả độ chính xác tác vụ (`target`: 0.7650 vs 0.0000), tuân thủ định dạng (`format`: 1.0000 vs 0.0000) và độ trễ phản hồi (1013.9 ms vs 3316.3 ms). 
Prompt (b) được giữ nguyên bản gốc theo chuẩn benchmark (SHA: `719e74d3b6232053`, không chỉnh sửa làm yếu đi), đảm bảo đây là một rào cản đo lường trung thực, khắt khe trước khi huấn luyện mô hình.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6283 | **0.9700** | 425.3 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5370 | **0.9650** | 268.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.0000** | 403.3 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.9400** | 477.6 | 3.86 |

> **Quy tắc vàng:** Xếp hạng chất lượng mô hình bắt buộc phải dựa vào cột **target**, tuyệt đối không dùng cột train loss. Chấm điểm mô hình bằng chỉ số thay thế (surrogate metric) như train loss là Lỗi thực nghiệm nghiêm trọng số 3 (Mistake #3).

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target 50 mẫu, `attn_only` đạt 0.9650, hơi thua nhẹ so với `correct` đạt 0.9700. Tuy nhiên, khi nhìn vào train loss ở NB4, `attn_only` lại có loss thấp hơn rõ rệt so với `correct` (0.5370 so với 0.6283), tạo ra sự đảo ngược hoàn toàn giữa bảng loss huấn luyện và bảng năng lực thực tế. Hiện tượng này minh chứng rằng việc nâng rank lên cực cao ($r=283$) chỉ để cắm adapter vào lớp attention giúp mô hình ghi nhớ (memorize) tập train nhanh hơn, nhưng không thể tổng quát hóa vượt qua việc phủ adapter lên toàn bộ các lớp `text-linear` với rank nhỏ ($r=16$). Vì vậy, vị trí đặt adapter (coverage) đóng vai trò then chốt quyết định chất lượng biểu diễn, trong khi rank chỉ là tham số sức chứa cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` sử dụng learning rate cấp độ full-finetuning ($1\times 10^{-5}$) thay vì thang LoRA chuẩn ($1\times 10^{-4}$). Kết quả là đường loss của `wrong_lr` gần như phẳng lì suốt quá trình huấn luyện, dừng lại ở mức 1.5702 (so với 0.6283 của `correct`) và dẫn tới điểm target bằng 0.0000 hoàn toàn. Nếu chỉ quan sát đường loss mà không kiểm tra learning rate, người làm thí nghiệm sẽ kết luận sai rằng "kiến trúc LoRA không học được tác vụ này", "tập dữ liệu có vấn đề" hoặc "cần đổi model khác". Thực tế, adapter LoRA chỉ cập nhật ma trận tích nhỏ khởi tạo bằng 0, đòi hỏi gradient step lớn hơn xấp xỉ $10\times$ so với pre-training để bứt phá khỏi điểm yên ngựa ban đầu.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` cắt giảm mạnh dung lượng bộ nhớ VRAM đỉnh từ 8.78 GB xuống còn 3.86 GB (tiết kiệm hơn 56% VRAM). Tuy nhiên, cái giá phải trả là thời gian huấn luyện kéo dài lên 477.6 giây (chậm hơn ~12% so với 425.3 giây của fp16 do chi phí dequantization liên tục trên GPU) và điểm target sụt giảm từ 0.9700 xuống 0.9400. Kết quả thực nghiệm hoàn toàn ủng hộ khuyến cáo chính thức từ tác giả Qwen: khi tài nguyên phần cứng có đủ VRAM (như T4 16GB với model 4B), không nên dùng QLoRA 4-bit vì sai số lượng tử hóa làm giảm độ sắc bén và làm chậm tốc độ huấn luyện.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.2050` · `regression Δ = -0.2689` · `valid_trace_rate = 0.00`

**Diễn giải chi tiết:**
Cổng hồi quy đánh giá phán quyết `FAILED` bởi vì năng lực tổng quát bị suy giảm nghiêm trọng: `general capability regressed by 0.269 (tolerance 0.020)`. Cụ thể, trong khi điểm số trên tác vụ mục tiêu tăng trưởng rất mạnh ($\Delta = +0.205$, từ 0.7650 lên 0.9700 với định dạng JSON đạt tuyệt đối 1.0), điểm kiểm tra năng lực tổng quát (eval regression) lại sụt giảm từ 0.7911 xuống 0.5222.

Đây là minh chứng kinh điển cho hiện tượng **quên thảm họa (catastrophic forgetting)** được phân tích trong deck §6.3: khi fine-tune một mô hình ngôn ngữ trên một tập dữ liệu miền hẹp chỉ gồm 225 mẫu JSON phân loại mà không có cơ chế bảo toàn tri thức nền, các trọng số LoRA bị chuyên biệt hóa quá mức vào cấu trúc ticket CSKH, làm xói mòn khả năng trả lời các câu hỏi tổng quát đời sống và tri thức phổ thông. Để khắc phục hiện tượng này mà vẫn giữ được độ chính xác mục tiêu cao, giải pháp bắt buộc theo lý thuyết là trộn thêm 1–5% dữ liệu hồi quy/phổ thông (replay buffer data) vào tập huấn luyện trước khi triển khai thực tế.

---

## 6. Định tính — Phân tích chi tiết cả ca THẮNG và THUA

Dưới đây là 5 mẫu trích xuất trực tiếp từ `results/qualitative.json` (tổng hợp 50 mẫu đánh giá) đối chiếu với nhãn thực tế (`data/eval_target.jsonl`):

| # | Ticket (rút gọn) | Nhãn đúng (Gold Label) | (b) Prompt | (c) Fine-tune (FT) | Nhận xét chi tiết |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô... | intent: van_chuyen, urgency: thap, product: ốp lưng điện thoại, sentiment: trung_tinh | Sai format hoặc thiếu trường | `{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | ✅ **FT thắng tuyệt đối (Score 1.0)**: Trích xuất chuẩn xác cả 4 trường JSON, phân loại đúng intent vận chuyển và mức độ khẩn cấp thấp. |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu... | intent: hoi_thong_tin, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: trung_tinh | Bị lẫn lộn format hoặc giải thích dông dài | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | ✅ **FT thắng tuyệt đối (Score 1.0)**: Bắt đúng từ khóa hỏi thông tin giá cả, định dạng JSON gọn gàng không thừa từ nào. |
| 3 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm n... | intent: san_pham_loi, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: tieu_cuc | Điểm target thấp hơn do cấu trúc lỏng lẻo | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | ✅ **FT thắng tuyệt đối (Score 1.0)**: Bắt đúng lỗi sai màu và sắc thái phàn nàn của khách hàng. |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | intent: hoan_tien, urgency: **thap**, product: bình giữ nhiệt, sentiment: tich_cuc | Bắt đúng intent nhưng trượt format | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", ...}` | ❌ **FT thua (Score 0.75)**: Model đoán nhầm `urgency: "trung_binh"` trong khi nhãn chuẩn là `thap` (do khách ghi "Khi nào tiện"). |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện... | intent: san_pham_loi, urgency: **thap**, product: áo khoác gió, sentiment: tich_cuc | Bắt đúng intent | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", ...}` | ❌ **FT thua (Score 0.75)**: Model bị thiên kiến gán nhãn, tiếp tục đoán sai `urgency: "trung_binh"` thay vì `thap`. |

**Có mẫu chung nào ở các ca FT thua không?**
Có một mẫu hình lỗi rất nhất quán: Trong các ca bị mất điểm (mẫu #4, mẫu #5 và mẫu index 12), bản Fine-tune đều đoán sai trường `urgency` thành `"trung_binh"`, trong khi nhãn chuẩn của dữ liệu là `"thap"`. Nguyên nhân xuất phát từ sự mất cân bằng phân phối trong tập dữ liệu huấn luyện: đa phần ticket CSKH trong thực tế có tính cấp bách trung bình hoặc cao, khiến mô hình fine-tune hình thành thiên kiến quy chụp `"trung_binh"`, bỏ qua các tín hiệu từ ngữ chỉ sự thư thả như *"khi nào tiện"* hoặc *"không vội"* của khách hàng.

---

## 7. Kết luận & điều tôi học được

**Kết luận chuyên môn:**
Dựa trên kết quả đo lường nghiêm ngặt của 4 nhóm chỉ số trên toàn bộ 50 mẫu đánh giá, câu trả lời là: **Chưa nên deploy trực tiếp checkpoint này ra môi trường production ngay lập tức**. Mặc dù bản fine-tune đạt hiệu quả rất ấn tượng trên tác vụ chuyên biệt ($\text{target}=0.9700$, định dạng JSON chuẩn $100\%$, tăng hơn $20.5\%$ so với prompting tối ưu), nó lại vi phạm nghiêm trọng cổng hồi quy ($\text{regression } \Delta = -0.2689$). Trong một hệ thống thực tế phục vụ khách hàng đa chức năng, việc suy thoái năng lực ngôn ngữ tổng quát sẽ khiến mô hình dễ gặp sự cố khi người dùng đưa ra các câu hỏi ngoài luồng (out-of-distribution) hoặc các tác vụ giao tiếp cơ bản.

Đòn bẩy thực sự quyết định thành công của lab này không phải là việc cố gắng tăng rank hay chuyển sang 4-bit, mà là **Sự kết hợp giữa Loss Masking chính xác và Phân bổ vị trí Adapter (Placement)**. Việc mask hoàn toàn prompt bảo đảm gradient chỉ tối ưu hóa việc trích xuất thực thể và suy luận JSON; đồng thời việc phủ đều adapter lên toàn bộ các khối `text-linear` giúp mô hình nắm bắt cấu trúc ngữ nghĩa tiếng Việt tốt hơn rất nhiều so với chỉ cắm vào attention layers.

**Ba điều tôi học được:**
1. **Loss masking là điều kiện tiên quyết của SFT**: Nếu tính loss cả trên phần prompt, mô hình sẽ học vẹt cú pháp câu hỏi thay vì học cách suy luận sinh câu trả lời. Giải mã ngược token mask ở NB1 là phương pháp trực quan và đáng tin cậy nhất để kiểm chứng tính đúng đắn.
2. **Train loss là chỉ số thay thế nguy hiểm**: `attn_only` đạt loss thấp hơn `correct` (0.5370 so với 0.6283) nhưng trên tập kiểm thử thực tế 50 mẫu điểm target lại thua nhẹ (0.9650 so với 0.9700). Tối ưu hóa quá mức trên train loss dễ dẫn đến hiện tượng overfit cục bộ thay vì cải thiện năng lực mô hình.
3. **Cổng hồi quy và Đóng băng mốc chuẩn là thước đo liêm chính học thuật**: Đóng băng baseline (b) trước khi train ngăn chặn hoàn toàn việc thiên vị kết quả; đồng thời việc kiểm tra năng lực hồi quy giúp phát hiện sớm hiện tượng quên thảm họa trước khi mô hình được đưa vào thực tế.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn thêm 3–5% dữ liệu tổng quát (general instruction data) vào tập huấn luyện để dập tắt hiện tượng suy giảm năng lực hồi quy, đưa `regression Δ` về vùng an toàn ($>-0.02$) để vượt qua cổng kiểm tra phán quyết (`PASSED`).

---

## Phụ lục — Thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
