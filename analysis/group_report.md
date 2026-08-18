# Group Report — Lab 18: Production RAG

**Nhóm:** K34 - Day 18  
**Học viên:** Trần Chí Vũ (MSSV: 2A202601044)  
**Ngày thực hiện:** 18/08/2026

---

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Trần Chí Vũ | M1: Advanced Chunking | ☑ | 13/13 |
| Trần Chí Vũ | M2: Hybrid Search | ☑ | 5/5 |
| Trần Chí Vũ | M3: Reranking | ☑ | 5/5 |
| Trần Chí Vũ | M4: Evaluation | ☑ | 4/4 |
| Trần Chí Vũ | M5: Enrichment Pipeline | ☑ | 10/10 |

---

## Kết quả RAGAS

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 1.0000 | +1.0000 |
| Answer Relevancy | 0.1300 | 0.4007 | +0.2707 |
| Context Precision | 0.0000 | 0.6911 | +0.6911 |
| Context Recall | 0.0000 | 0.6432 | +0.6432 |

---

## Key Findings

1. **Biggest improvement:**  
   Chỉ số **Faithfulness** tăng từ 0.0 lên tuyệt đối 1.0000 nhờ hệ thống Hierarchical Chunking kết hợp Contextual Enrichment giúp LLM luôn bám sát ngữ cảnh thực tế của tài liệu và không bị hallucination. **Context Precision** và **Context Recall** tăng vượt bậc (từ 0.0000 lên ~0.69 và ~0.64) nhờ sự kết hợp giữa BM25 tiếng Việt (underthesea segmentation) và Dense Search (Qdrant + bge-m3) qua cơ chế RRF (Reciprocal Rank Fusion).

2. **Biggest challenge:**  
   Xử lý các câu hỏi phức tạp mang tính **Multi-hop / Multi-constraint** (ví dụ câu hỏi vừa yêu cầu tỷ lệ phần trăm thử việc vừa yêu cầu mức lương theo cấp bậc Junior). Khi đó câu trả lời nằm ở 2 tài liệu độc lập (`thu_viec.md` và `bang_luong_2024.md`), việc tìm kiếm đơn luồng có xu hướng chỉ lấy được 1 trong 2 tài liệu.

3. **Surprise finding:**  
   Kỹ thuật **Contextual Prepend** và **HyQA** trong Module 5 tạo ra sự khác biệt rất lớn đối với khả năng match keyword của BM25. Khi các chunk ngắn được bổ sung tiêu đề ngữ cảnh nguồn tài liệu, tỷ lệ lọc trúng tài liệu liên quan tăng lên rõ rệt mà không làm tăng độ trễ truy vấn.

---

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):**
   - Naive RAG gặp thất bại hoàn toàn ở cả 4 chỉ số do cắt đoạn thô sơ theo paragraph và tìm kiếm dense-only không qua rerank.
   - Production RAG Pipeline đạt điểm tuyệt đối về độ trung thực (Faithfulness 1.0), Context Precision đạt 0.6911 và Recall đạt 0.6432.

2. **Biggest win — module nào, tại sao:**
   - **M1 (Hierarchical Chunking) + M2 (Hybrid Search BM25 + Dense + RRF):** Đảm bảo child chunks có độ dài ngắn gọn (256 ký tự) giúp tăng độ chính xác truy xuất (precision), đồng thời parent context (2048 ký tự) cung cấp đầy đủ ngữ cảnh để LLM sinh câu trả lời chính xác.

3. **Case study — 1 failure, Error Tree walkthrough:**
   - *Câu hỏi:* "Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"
   - *Walkthrough:* Output thiếu số tiền 17 triệu $\to$ Context thiếu file `bang_luong_2024.md` $\to$ Nguyên nhân là câu hỏi đòi hỏi thông tin từ 2 file khác nhau $\to$ Giải pháp: Áp dụng Query Decomposition để tách câu hỏi thành 2 sub-queries trước khi retrieval.

4. **Next optimization nếu có thêm 1 giờ:**
   - Triển khai **Query Routing & Decomposition** cho các câu hỏi đa bước (multi-hop reasoning).
   - Thiết lập **Metadata Filtering** tự động theo phiên bản chính sách mới nhất (`v2024` > `v2023`, `v2.0` > `v1.0`).
   - Tối ưu hóa trọng số Hybrid Search ($w_{bm25} : w_{dense}$) theo từng loại câu hỏi (factoid vs descriptive).
