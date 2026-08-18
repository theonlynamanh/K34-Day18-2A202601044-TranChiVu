# Individual Reflection — Lab 18: Production RAG Pipeline

**Họ và tên:** Trần Chí Vũ  
**Mã số sinh viên:** 2A202601044  
**Lớp:** K34  
**Module phụ trách:** Toàn bộ Pipeline (M1, M2, M3, M4, M5)

---

## Phần 1: Mapping Bài Giảng vào Thực Tế Triển Khai

| Lecture Concept | Module | Hàm / Lớp cụ thể | Observation & Phân tích thực tế |
|---|---|---|---|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Dùng cosine similarity giữa các câu liên tiếp (`threshold = 0.85`) với mô hình `all-MiniLM-L6-v2`. Giúp bảo toàn trọn vẹn ngữ nghĩa câu, tránh tình trạng ngắt đoạn thô giữa chừng như basic chunking. |
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | Tạo cấu trúc Parent (2048 chars) và Child (256 chars) có liên kết `parent_id`. Khi tìm kiếm truy xuất child chunk đạt độ chính xác cao (Precision), khi trả context cho LLM nạp parent chunk giúp đầy đủ ngữ cảnh (Recall). |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Parse markdown headers (`#`, `##`, `###`) để chia chunk theo từng section tài liệu, lưu giữ tiêu đề vào metadata giúp phân loại nội dung chính xác. |
| **Vietnamese Word Segmentation** | M2 | `segment_vietnamese()` | Sử dụng `underthesea.word_tokenize` và chuẩn hóa dấu gạch dưới `_` thành khoảng trắng `" "` để tương thích hoàn hảo với tokenizer của BM25. |
| **BM25 + Dense Fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | Kết hợp BM25 (từ khóa chính xác) và Dense Search (vector ngữ nghĩa với Qdrant và bge-m3) bằng công thức $score(d) = \sum \frac{1}{k + rank + 1}$ với $k=60$, giải quyết triệt để điểm yếu của dense-only khi gặp từ khóa chuyên ngành. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Tái xếp hạng top-20 ứng viên xuống top-3 bằng mô hình `BAAI/bge-reranker-v2-m3`. Đánh giá tương tác chéo giữa câu hỏi và tài liệu, lọc bỏ các chunk chỉ trùng lặp từ khóa nhưng không liên quan về ngữ nghĩa. |
| **RAGAS 4 Metrics** | M4 | `evaluate_ragas()` | Đánh giá toàn diện 4 trụ cột: `faithfulness` (độ trung thực), `answer_relevancy` (độ liên quan của câu trả lời), `context_precision` (độ chính xác ngữ cảnh) và `context_recall` (độ bao phủ thông tin). |
| **Diagnostic Failure Tree** | M4 | `failure_analysis()` | Tự động phân loại lỗi theo cây chẩn đoán (Diagnostic Tree) để tìm nguyên nhân gốc rễ và đề xuất giải pháp sửa chữa cụ thể cho từng câu hỏi trong Bottom-5. |
| **Enrichment & Contextual Prepend** | M5 | `contextual_prepend()`, `_enrich_single_call()` | Bổ sung ngữ cảnh xuất xứ tài liệu, tạo câu hỏi giả thuyết (HyQA) và trích xuất metadata tự động. Sử dụng chế độ combined single-call để tối ưu chi phí và tốc độ xử lý. |

---

## Phần 2: Khó Khăn Gặp Phải & Cách Giải Quyết

1. **Lỗi `UnicodeEncodeError: 'charmap' codec can't encode character` trên Windows Console:**
   - *Exact Error Message:* `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4cc' in position 2: character maps to <undefined>`
   - *Nguyên nhân:* Môi trường dòng lệnh Windows mặc định dùng bảng mã cp1252/cp437 không hỗ trợ in trực tiếp các emoji UTF-8.
   - *Cách giải quyết:* Bổ sung cấu hình `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` ngay đầu các entry points (`main.py`, `pipeline.py`, `check_lab.py`, `naive_baseline.py`).

2. **Xung đột phiên bản tài liệu (Outdated vs Current Version):**
   - *Hiện tượng:* Retriever lấy nhầm tài liệu `mat_khau_v1.md` (yêu cầu 8 ký tự, 90 ngày) thay vì `mat_khau_v2.md` (12 ký tự, 120 ngày).
   - *Cách xử lý:* Gắn metadata `version` và sử dụng Contextual Prepend nêu rõ tên tệp tin và trạng thái tài liệu để LLM nhận biết và ưu tiên phiên bản mới nhất.

3. **Multi-hop Retrieval & Missing Context:**
   - *Hiện tượng:* Với câu hỏi tính lương thử việc cấp Junior, retriever chỉ lấy được file `thu_viec.md` mà thiếu file `bang_luong_2024.md`.
   - *Cách xử lý:* Phân tích nguyên nhân qua Diagnostic Tree, đề xuất phương án Query Decomposition để tách câu hỏi phức thành các truy vấn đơn lẻ trước khi tổng hợp kết quả.

---

## Phần 3: Action Plan Áp Dụng Cho Project Cá Nhân

### Project: Hệ Thống Trợ Lý Tra Cứu Tài Liệu Pháp Lý & Quy Chế Nội Bộ Doanh Nghiệp

### Hiện tại
- **RAG Pipeline hiện tại:** Sử dụng naive chunking theo số lượng token cố định (512 tokens), lưu trữ trên FAISS dense vector search đơn thuần, chưa có reranking và chưa có đánh giá chuẩn hóa.
- **Known Issues:** 
  - Thường xuyên bị đứt đoạn điều khoản pháp luật khi cắt ngang văn bản.
  - Tỷ lệ tìm kiếm sai số liệu/mã số văn bản cao do dense vector không bắt chuẩn xác số hiệu (ví dụ: "Nghị định 13/2023/NĐ-CP").
  - Không có cơ chế phát hiện tài liệu đã hết hiệu lực thi hành.

### Kế hoạch áp dụng các kỹ thuật từ Lab 18
1. **Chunking Strategy:**
   - Sử dụng **Structure-Aware Chunking** kết hợp **Hierarchical Chunking**: Tách theo từng Điều/Khoản trong văn bản pháp luật, giữ nguyên cấu trúc phân cấp Chương $\to$ Mục $\to$ Điều $\to$ Khoản.
2. **Search Architecture:**
   - Triển khai **Hybrid Search** kết hợp `underthesea` word segmentation cho tiếng Việt + BM25Okapi + Dense Vector (bge-m3) qua Reciprocal Rank Fusion (RRF với $k=60$).
3. **Reranking Layer:**
   - Tích hợp `CrossEncoderReranker` với mô hình `bge-reranker-v2-m3` (top-25 xuống top-5) để loại bỏ nhiễu trước khi đưa vào LLM Context Window.
4. **Evaluation Framework:**
   - Xây dựng bộ test set 50 câu hỏi tiêu chuẩn đa độ khó, định kỳ chạy **RAGAS Evaluation** đo lường 4 chỉ số (mục tiêu tất cả các chỉ số $\ge 0.80$).
5. **Enrichment:**
   - Áp dụng **Contextual Prepend** để gắn tên văn bản, cơ quan ban hành, ngày có hiệu lực vào đầu mỗi chunk trước khi index.

### Timeline Triển Khai
- **Tuần 1:** Thiết kế lại bộ tiền xử lý dữ liệu và Structure-Aware Chunking cho toàn bộ kho văn bản quy chế.
- **Tuần 2:** Thiết lập cụm Qdrant vector database và tích hợp BM25 Hybrid Search với RRF.
- **Tuần 3:** Tích hợp Cross-Encoder Reranker, kiểm thử độ trễ (latency breakdown report) và tối ưu hóa thời gian phản hồi.
- **Tuần 4:** Xây dựng dashboard theo dõi chỉ số RAGAS và tự động cảnh báo câu hỏi có điểm Faithfulness/Recall thấp qua Diagnostic Tree.

---

## Phần 4: Đánh Giá Đóng Góp Kỹ Thuật

- **Các module đã implement:** M1 (Chunking), M2 (Search), M3 (Reranking), M4 (Eval), M5 (Enrichment), Pipeline End-to-End.
- **Số tests pass:** 37/37 (100%).
- **TODO Markers:** 0 TODOs remaining.
- **Tự chấm điểm:** 10/10 (Đạt toàn bộ tiêu chí cốt lõi và các tiêu chí thưởng bonus).
