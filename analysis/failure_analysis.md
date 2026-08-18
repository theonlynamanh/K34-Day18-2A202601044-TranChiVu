# Failure Analysis — Lab 18: Production RAG

**Nhóm:** K34 - Day 18  
**Học viên thực hiện:** Trần Chí Vũ (MSSV: 2A202601044)  
**Phân công modules:** Trần Chí Vũ (M1: Chunking, M2: Hybrid Search, M3: Reranking, M4: Evaluation, M5: Enrichment)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 1.0000 | +1.0000 |
| Answer Relevancy | 0.1300 | 0.4007 | +0.2707 |
| Context Precision | 0.0000 | 0.6911 | +0.6911 |
| Context Recall | 0.0000 | 0.6432 | +0.6432 |

---

## Bottom-5 Failures

### #1
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Trích từ `thu_viec.md`. Nhân viên thử việc được nhận 85% mức lương của cấp bậc tương ứng theo bảng lương công ty. Phụ cấp ăn trưa được áp dụng đầy đủ từ ngày đầu tiên.
- **Worst metric:** `answer_relevancy` (0.2275)
- **Error Tree:** Output chưa tính con số cụ thể → Context lấy được quy định 85% từ `thu_viec.md` nhưng thiếu bảng lương `bang_luong_2024.md` (mức lương Junior max: 20tr) → Query retriever chỉ tìm đơn lẻ file thử việc.
- **Root cause:** Multi-hop reasoning / multi-document retrieval gap. Hệ thống chưa liên kết được giữa tài liệu điều kiện và bảng tra cứu mức lương.
- **Suggested fix:** Áp dụng Query Decomposition hoặc Multi-hop Retrieval (Sub-query 1: "Mức lương tối đa Junior", Sub-query 2: "Tỷ lệ lương thử việc") rồi tổng hợp context trước khi sinh câu trả lời.

---

### #2
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Trích từ `nghi_phep_dac_biet.md`. Các trường hợp được nghỉ phép đặc biệt có lương: kết hôn 3 ngày, tang lễ 3 ngày, sinh con 5 ngày...
- **Worst metric:** `answer_relevancy` (0.1936) / `context_precision` (0.6000)
- **Error Tree:** Output trả lời lệch chủ đề sang "phép đặc biệt" → Context retrieval bị nhiễu do từ khóa "nghỉ phép" xuất hiện nhiều trong `nghi_phep_dac_biet.md` hơn là `nghi_phep_nam_v2024.md`.
- **Root cause:** Keyword overlap ambiguity. Từ khóa "nghỉ phép" quá chung chung khiến BM25 và Dense Search lấy nhầm đoạn văn liên quan đến các loại nghỉ phép khác.
- **Suggested fix:** Cải thiện cấu trúc token BM25 với N-gram ("nghỉ phép năm") hoặc thêm Metadata Prepend để phân biệt rõ `category: "annual_leave"` và `category: "special_leave"`.

---

### #3
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Trích từ `mat_khau_v1.md`. Mật khẩu phải có tối thiểu 8 ký tự, bao gồm ít nhất 1 chữ hoa, 1 chữ thường và 1 số...
- **Worst metric:** `answer_relevancy` (0.2167)
- **Error Tree:** Output lấy thông tin từ phiên bản tài liệu cũ v1.0 → Retrieval lấy cả 2 file `mat_khau_v1.md` và `mat_khau_v2.md` nhưng reranker không ưu tiên tài liệu mới nhất.
- **Root cause:** Temporal conflict & version precedence. Hệ thống chưa có cơ chế lọc tài liệu hết hiệu lực (deprecated / superseded).
- **Suggested fix:** Metadata filtering theo `version` hoặc gán thuộc tính `is_active: true/false` vào chunk metadata, kết hợp metadata filtering trong bước retrieval.

---

### #4
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG. Nhân viên thử việc KHÔNG được nghỉ phép năm. Nếu cần nghỉ, phải xin nghỉ không lương và được trưởng phòng phê duyệt.
- **Got:** Trích từ `nghi_phep_dac_biet.md`. Nhân viên được nghỉ có lương trong các trường hợp kết hôn, tang lễ...
- **Worst metric:** `answer_relevancy` (0.2489)
- **Error Tree:** Output trả lời về nghỉ phép đặc biệt → Retrieval ưu tiên từ khóa "nghỉ phép" thay vì thực thể chính "nhân viên thử việc".
- **Root cause:** Entity weighting imbalance. BM25 và Vector embedding cho trọng số cao vào từ "nghỉ phép" (tần suất cao trong toàn bộ corpus) thay vì "thử việc" (từ khóa đặc thù).
- **Suggested fix:** Sử dụng BM25 với dynamic field boosting (ví dụ boost `source` hoặc `metadata.topic`) hoặc dùng HyQA (sinh câu hỏi giả thuyết "Nhân viên thử việc có được nghỉ phép năm không?" gắn vào chunk của `thu_viec.md`).

---

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trích từ `nghi_phep_khong_luong.md`. Điều kiện áp dụng nghỉ phép không lương...
- **Worst metric:** `context_recall` (0.4000)
- **Error Tree:** Context thiếu cả 2 thông tin cốt lõi (chính sách thâm niên 2024 và bảng lương Senior) → Retrieval chỉ tìm thấy file nghỉ phép không lương.
- **Root cause:** Complex multi-constraint query. Câu hỏi chứa 3 điều kiện: chức danh Senior, thâm niên 9 năm, số ngày phép năm + mức lương.
- **Suggested fix:** Phân tích câu hỏi phức tạp thành sub-queries độc lập:
  1. `Query 1:` "Quy định cộng ngày phép theo thâm niên chính sách v2024"
  2. `Query 2:` "Khung lương cấp bậc Senior P3 P4"

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
`"Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"`

**Error Tree walkthrough:**
1. **Output đúng?** $\to$ **KHÔNG**. Output chỉ nêu mức 85% chung chung, không tính ra con số 17.000.000 VNĐ.
2. **Context đúng?** $\to$ **MỘT NỬA**. Context chỉ chứa `thu_viec.md` (nêu tỷ lệ 85%), hoàn toàn thiếu `bang_luong_2024.md` (nêu mức lương Junior tối đa là 20.000.000 VNĐ).
3. **Query rewrite OK?** $\to$ Query ban đầu là câu hỏi phức, chưa được rewrite/decompose để tìm đồng thời cả điều kiện lương thử việc và bảng lương cơ bản.
4. **Fix ở bước:**
   - **Bước 1 (Enrichment - M5):** Tăng cường HyQA cho chunk bảng lương, sinh câu hỏi liên quan đến lương thử việc cấp bậc.
   - **Bước 2 (Search - M2):** Áp dụng Query Expansion / Multi-query generation để truy xuất song song nhiều nguồn tài liệu liên quan.
   - **Bước 3 (Generation):** Cải thiện prompt yêu cầu LLM kết hợp dữ liệu giữa các context để tính toán số học chính xác.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Triển khai Query Decomposition Agent:** Tự động tách câu hỏi đa phần (multi-hop) thành các sub-queries độc lập trước khi gửi vào Search Engine.
- **Bổ sung Metadata Filter theo Version:** Đánh dấu tài liệu `v2023`, `v1.0` là deprecated, ưu tiên tài liệu `v2024`, `v2.0` khi cùng chủ đề để loại bỏ triệt để xung đột phiên bản.
- **Tích hợp FlashRank/CrossEncoder tối ưu Top-K:** Điều chỉnh ngưỡng RRF và số lượng chunks trả về cho LLM (Top-5 thay vì Top-3 đối với câu hỏi phức hợp) để tăng Context Recall lên trên 85%.
