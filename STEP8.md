# Phân tích kết quả benchmark

Các số liệu dưới đây được đo bằng `python src/benchmark.py` ở chế độ offline. Recall là điểm trung bình theo câu hỏi, chấm `0`, `0.5` hoặc `1` dựa trên các chuỗi `expected_contains` xuất hiện trong câu trả lời.

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|
|z Standard | Baseline | 250 | 375 | 0.000 | 0 | 0 |
| Standard | Advanced | 234 | 414 | 0.214 | 2375 | 0 |
| Long-context stress | Baseline | 396 | 2688 | 0.000 | 0 | 0 |
| Long-context stress | Advanced | 530 | 1963 | 0.333 | 294 | 1 |

## 1. Recall qua thread mới

**Số liệu:** Standard đạt `0.214` với Advanced so với `0.000` với Baseline; stress đạt `0.333` so với `0.000`. Advanced có lợi thế recall ở cả hai bộ, nhưng điểm tuyệt đối vẫn thấp, nhất là stress.

**Cơ chế:** Benchmark huấn luyện trong một thread rồi hỏi recall ở thread mới. Advanced gọi `extract_profile_updates()` trong `src/memory_store.py`, ghi facts vào `User.md`, sau đó `_offline_response()` trong `src/agent_advanced.py` đọc facts để tạo câu trả lời. Baseline không có profile bền vững nên không mang facts từ thread huấn luyện sang thread recall.

**Giới hạn:** Bộ trích xuất và truy xuất chỉ hỗ trợ một số field và từ khóa; câu hỏi tổng hợp hoặc cách diễn đạt khác dễ làm thiếu fact. Chấm điểm theo chuỗi con cũng không hiểu từ đồng nghĩa, nên các số này chứng minh Advanced tốt hơn Baseline trên bộ test hiện tại, không chứng minh recall đã đáng tin cậy cho mọi câu hỏi.

## 2. Chi phí ở hội thoại ngắn

**Số liệu:** Trong Standard, Advanced dùng `414` prompt tokens so với `375` của Baseline, nhiều hơn `39` (khoảng `10.4%`). Tuy nhiên, `Agent tokens only` của Advanced là `234`, thấp hơn Baseline `250`; số đo này không cho thấy Advanced sinh nhiều token trả lời hơn.

**Cơ chế:** Khi ước lượng prompt, Advanced cộng token của profile `User.md`, summary và các message gần nhất. Cập nhật fact còn có chi phí ghi profile. Standard có `0` compactions, nên chưa có lợi ích nén để bù phần context bổ sung.

**Giới hạn:** Kết quả ủng hộ nhận định Advanced tốn prompt context hơn ở hội thoại ngắn, nhưng không ủng hộ nhận định nó tốn hơn ở `Agent tokens only`. Không nên gộp hai cột này thành một khẳng định chung về “token cost”.

## 3. Tác động của compact memory

**Số liệu:** Trong stress benchmark, `Prompt tokens processed` của Advanced là `1963`, thấp hơn Baseline `2688` đúng `725` (khoảng `27.0%`). Advanced có `1` compaction, Baseline có `0`. Ngược lại, `Agent tokens only` của Advanced là `530`, cao hơn Baseline `396`.

**Cơ chế:** `CompactMemoryManager` trong `src/memory_store.py` giữ các message gần nhất và tóm tắt phần cũ khi vượt ngưỡng. Advanced ước lượng context từ profile, summary và message còn lại; vì thế lợi thế quan sát được nằm ở cột prompt context, không phải số token sinh ra trực tiếp.

**Giới hạn:** Stress run chỉ ghi nhận `1` compaction, nên đây là bằng chứng compact đã kích hoạt và giảm context ở phép đo hiện tại, chưa phải bằng chứng về hành vi ổn định qua nhiều lần compact. `Agent tokens only` tăng `134`, cho thấy compact không tự làm câu trả lời ngắn hơn.

## 4. Memory growth và rủi ro

**Số liệu:** Ở stress benchmark, Advanced ghi `294` bytes và `1` compaction; Baseline ghi `0` bytes và `0` compactions. File profile là chi phí lưu trữ thật mà Baseline không phát sinh.

**Cơ chế và rủi ro:** `UserProfileStore.upsert_fact()` cập nhật facts trong profile, còn bộ trích xuất dựa trên heuristic thay vì confidence score. Khi chạy thử, câu mục tiêu học tập “là xây được agent nhớ người dùng tốt hơn” từng bị nhận nhầm thành tên người dùng; điều này cho thấy một lượt nhiễu có thể được ghi thành fact và tồn tại hoặc ghi đè fact cũ cho tới khi được sửa. Vì vậy cần kiểm tra độ tin cậy, xử lý correction và cân nhắc decay/giới hạn kích thước memory trước khi dùng với dữ liệu dài hạn.

**Giới hạn phép đo:** Trong stress dataset chỉ có một conversation/user nên `294` bytes là kích thước profile được cộng một lần. Với Standard, `benchmark.py` cộng kích thước profile sau từng conversation cùng user; do đó `2375` là tổng các snapshot lặp, không phải kích thước cuối hay mức tăng ròng của một file. Ngoài ra, cột prompt hiện lấy giá trị context ở lượt cuối của mỗi conversation rồi tính trung bình, không cộng prompt cost của mọi lượt; cần giữ đúng cách hiểu này khi so sánh.