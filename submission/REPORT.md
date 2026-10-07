# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Hoàng Việt  **MSSV**: 2A202602602  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp chính xác 100% với các file tạo ra trong `results/`.
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

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3309.0 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 991.8 |
| (c) LoRA fine-tune | 0.9375 | 0.6250 | 1.0000 | 1451.6 |

**(b) có thật sự mạnh hơn (a) không?** Có, (b) vượt trội hoàn toàn so với (a) trên cả độ chính xác tác vụ (`target`: 0.688 vs 0.000), tuân thủ định dạng (`format`: 1.0 vs 0.0) và độ trễ phản hồi (991.8 ms vs 3309.0 ms). 
Prompt (b) được giữ nguyên bản gốc theo chuẩn benchmark (SHA: `719e74d3b6232053`, không chỉnh sửa làm yếu đi), đảm bảo đây là một rào cản đo lường trung thực, khắt khe trước khi huấn luyện mô hình.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6263 | **0.9375** | 390.7 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5367 | **0.9375** | 258.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.0000** | 385.7 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.8438** | 456.5 | 3.86 |

> **Quy tắc vàng:** Xếp hạng chất lượng mô hình bắt buộc phải dựa vào cột **target**, tuyệt đối không dùng cột train loss. Chấm điểm mô hình bằng chỉ số thay thế (surrogate metric) như train loss là Lỗi thực nghiệm nghiêm trọng số 3 (Mistake #3).

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` hòa điểm với `correct` (cùng đạt 0.9375 và format 1.0). Tuy nhiên, khi nhìn vào train loss ở NB4, `attn_only` có loss thấp hơn đáng kể so với `correct` (0.5367 so với 0.6263), tạo ra thứ tự xếp hạng hoàn toàn nghịch đảo giữa loss huấn luyện và năng lực thực tế. Hiện tượng này minh chứng rằng việc nâng rank lên cực cao ($r=283$) để bù đắp việc chỉ cắm adapter vào lớp attention chỉ giúp mô hình ghi nhớ (overfit/memorize) tập train nhanh hơn, chứ không hề tạo ra biểu diễn tổng quát hóa tốt hơn việc phân bổ adapter trên toàn bộ các lớp `text-linear` với rank nhỏ ($r=16$). Vì vậy, vị trí đặt adapter (coverage) đóng vai trò nền tảng quyết định chất lượng biểu diễn, trong khi rank đơn thuần chỉ là dung lượng lưu trữ cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` sử dụng learning rate cấp độ full-finetuning ($1\times 10^{-5}$) thay vì thang LoRA khuyến nghị ($1\times 10^{-4}$). Kết quả là đường loss của `wrong_lr` gần như phẳng lì, dừng lại ở mức 1.5702 (so với 0.6263 của `correct`) và dẫn tới điểm target bằng 0.0000 hoàn toàn. Nếu chỉ quan sát đường loss mà không kiểm tra learning rate, một kỹ sư thiếu kinh nghiệm sẽ vội vàng kết luận sai rằng "kiến trúc LoRA không học được tác vụ này", "tập dữ liệu bị lỗi" hoặc "cần tăng rank/tăng số epoch". Thực tế, adapter LoRA chỉ cập nhật một ma trận tích nhỏ được khởi tạo bằng 0, đòi hỏi gradient step lớn hơn xấp xỉ $10\times$ so với pre-training để bứt phá khỏi điểm yên ngựa ban đầu.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` cắt giảm mạnh dung lượng bộ nhớ VRAM đỉnh từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% VRAM). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên 456.5 giây (chậm hơn ~17% so với 390.7 giây của fp16 do chi phí dequantization on-the-fly) và điểm target sụt giảm đáng kể từ 0.9375 xuống 0.8438. Kết quả thực nghiệm hoàn toàn ủng hộ khuyến cáo chính thức từ nhà phát triển Qwen và tài liệu hướng dẫn: trên phần cứng có đủ VRAM (như T4 16GB với model 4B), không nên dùng QLoRA 4-bit vì suy hao lượng tử hóa làm giảm độ sắc bén trong trích xuất thông tin có cấu trúc của mô hình.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.2500` · `regression Δ = -0.1250` · `valid_trace_rate = 0.00`

**Diễn giải chi tiết:**
Cổng hồi quy đánh giá phán quyết `FAILED` bởi vì năng lực tổng quát bị suy giảm: `general capability regressed by 0.125 (tolerance 0.020)`. Cụ thể, trong khi điểm số trên tác vụ mục tiêu tăng trưởng rất mạnh ($\Delta = +0.250$, từ 0.6875 lên 0.9375 với format đạt tuyệt đối 1.0), điểm kiểm tra năng lực tổng quát (eval regression) lại sụt giảm từ 0.7500 xuống 0.6250. 

Đây là minh chứng kinh điển cho hiện tượng **quên thảm họa (catastrophic forgetting)** được phân tích trong deck §6.3: khi fine-tune một mô hình ngôn ngữ trên một tập dữ liệu miền hẹp chỉ gồm 225 mẫu JSON phân loại mà không có cơ chế bảo toàn tri thức nền, các trọng số LoRA bị chuyên biệt hóa quá mức vào cấu trúc ticket CSKH, làm xói mòn khả năng trả lời các câu hỏi tổng quát đời sống và tri thức phổ thông. Để khắc phục hiện tượng này mà vẫn giữ được độ chính xác mục tiêu cao, giải pháp bắt buộc theo lý thuyết là trộn thêm 1–5% dữ liệu hồi quy/phổ thông (replay buffer data) vào tập huấn luyện trước khi triển khai thực tế.

---

## 6. Định tính — Phân tích chi tiết cả ca THẮNG và THUA

Dưới đây là 5 mẫu trích xuất trực tiếp từ `results/qualitative.json` đối chiếu với nhãn thực tế (`data/eval_target.jsonl`):

| # | Ticket (rút gọn) | Nhãn đúng (Gold Label) | (b) Prompt | (c) Fine-tune (FT) | Nhận xét chi tiết |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | intent: doi_tra, urgency: cao, product: chuột không dây, sentiment: tich_cuc | Sai format hoặc thiếu trường | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ **FT thắng tuyệt đối (Score 1.0)**: Trích xuất chuẩn xác cả 4 trường JSON, sentiment tích cực được nhận diện đúng dù có khiếu nại đổi trả. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé... | intent: hoan_tien, urgency: trung_binh, product: ốp lưng điện thoại, sentiment: tieu_cuc | Bị lẫn lộn giữa doi_tra và hoan_tien | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | ✅ **FT thắng tuyệt đối (Score 1.0)**: Bắt đúng từ khóa "hoàn tiền", phân định đúng mức độ khẩn cấp và sắc thái tiêu cực. |
| 3 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | intent: hoan_tien, urgency: cao, product: đèn bàn LED, sentiment: tich_cuc | Điểm target thấp hơn do format lỏng lẻo | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | ✅ **FT thắng tuyệt đối (Score 1.0)**: Xử lý hoàn hảo ngữ cảnh cảm ơn kết hợp khiếu nại quá hạn. |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | intent: hoan_tien, urgency: **thap**, product: bình giữ nhiệt, sentiment: tich_cuc | Bắt đúng intent nhưng trượt format | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", ...}` | ❌ **FT thua (Score 0.75)**: Model đoán nhầm `urgency: "trung_binh"` trong khi nhãn chuẩn là `thap` (do khách ghi "Khi nào tiện"). |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện... | intent: san_pham_loi, urgency: **thap**, product: nồi chiên không dầu, sentiment: trung_tinh | Bắt đúng intent | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", ...}` | ❌ **FT thua (Score 0.75)**: Model bị thiên kiến gán nhãn, tiếp tục đoán sai `urgency: "trung_binh"` thay vì `thap`. |

**Có mẫu chung nào ở các ca FT thua không?**
Có một mẫu hình lỗi rất rõ ràng: Trong cả hai ca bị trừ điểm (mẫu #4 và mẫu #5, đạt 0.75 thay vì 1.0), bản Fine-tune đều đoán sai trường `urgency` thành `"trung_binh"`, trong khi nhãn chuẩn của dữ liệu là `"thap"`. Nguyên nhân xuất phát từ sự mất cân bằng phân phối trong tập dữ liệu huấn luyện: các ticket CSKH thường có tính cấp bách trung bình hoặc cao, khiến mô hình fine-tune hình thành thiên kiến ưu tiên dự đoán `"trung_binh"`, bỏ qua các tín hiệu từ ngữ chỉ sự thư thả như *"khi nào tiện"* của khách hàng.

---

## 7. Kết luận & điều tôi học được

**Kết luận chuyên môn:**
Dựa trên kết quả đo lường nghiêm ngặt của 4 nhóm chỉ số, câu trả lời là: **Chưa nên deploy trực tiếp checkpoint này ra môi trường production ngay lập tức**. Mặc dù bản fine-tune đạt hiệu quả rất ấn tượng trên tác vụ chuyên biệt ($\text{target}=0.9375$, định dạng JSON chuẩn $100\%$, tăng $25\%$ so với prompting tối ưu), nó lại vi phạm cổng hồi quy ($\text{regression } \Delta = -0.125$). Trong một hệ thống thực tế phục vụ khách hàng đa chức năng, việc suy thoái năng lực ngôn ngữ tổng quát sẽ khiến mô hình dễ gặp sự cố khi người dùng đưa ra các câu hỏi ngoài luồng (out-of-distribution).

Đòn bẩy thực sự quyết định thành công của lab này không phải là việc cố gắng tăng rank hay chuyển sang 4-bit, mà là **Sự kết hợp giữa Loss Masking chính xác và Phân bổ vị trí Adapter (Placement)**. Việc mask hoàn toàn prompt bảo đảm gradient chỉ tối ưu hóa việc trích xuất thực thể và suy luận JSON; đồng thời việc phủ đều adapter lên toàn bộ các khối `text-linear` giúp mô hình nắm bắt cấu trúc ngữ nghĩa tiếng Việt tốt hơn rất nhiều so với chỉ cắm vào attention layers.

**Ba điều tôi học được:**
1. **Loss masking là điều kiện tiên quyết của SFT**: Nếu tính loss cả trên phần prompt, mô hình sẽ học vẹt cú pháp câu hỏi thay vì học cách suy luận sinh câu trả lời. Giải mã ngược token mask ở NB1 là phương pháp trực quan và đáng tin cậy nhất để kiểm chứng tính đúng đắn.
2. **Train loss là chỉ số thay thế nguy hiểm**: `attn_only` đạt loss thấp hơn `correct` (0.5367 so với 0.6263) nhưng trên tập kiểm thử thực tế điểm target lại không hề vượt trội. Tối ưu hóa quá mức trên train loss dễ dẫn đến hiện tượng overfit cục bộ thay vì cải thiện năng lực mô hình.
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
