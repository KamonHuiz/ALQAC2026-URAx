"""All Vietnamese prompt templates, as functions returning chat-message lists.

Kept in one place so prompts can be iterated/ablated without touching stage logic.
Label set (exact strings the grader expects):
    A_WIN | PARTIAL_A_WIN | PARTIAL_B_WIN | B_WIN
"""
from __future__ import annotations

LABELS = ["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]

LABEL_GLOSS = (
    "A_WIN = nguyên đơn (bên A) THẮNG hoàn toàn, được chấp nhận toàn bộ yêu cầu.\n"
    "PARTIAL_A_WIN = nguyên đơn thắng MỘT PHẦN LỚN (được chấp nhận trên 50% yêu cầu).\n"
    "PARTIAL_B_WIN = nguyên đơn chỉ thắng MỘT PHẦN NHỎ (được chấp nhận từ 50% trở xuống), bị đơn thắng phần lớn.\n"
    "B_WIN = bị đơn (bên B) THẮNG hoàn toàn, yêu cầu của nguyên đơn bị bác toàn bộ."
)


# --------------------------------------------------------------------------- #
# Stage 1 — case-specific retrieval query generation
# --------------------------------------------------------------------------- #
def query_generation(case_query: str, a_desc: str, b_desc: str, n: int) -> list[dict]:
    sys = ("Bạn là trợ lý truy hồi bằng chứng pháp lý. Nhiệm vụ: tạo các câu truy vấn "
           "tìm kiếm tiếng Việt (dạng từ khóa) để lấy ĐÚNG các đoạn nội dung quan trọng "
           "trong hồ sơ một bản án dân sự thông qua công cụ tìm kiếm BM25. "
           "Câu truy vấn nên cụ thể vào: tình tiết tranh chấp, số tiền/tài sản, nội dung "
           "hợp đồng/giao dịch, lời khai nhân chứng, kết quả định giá/thẩm định, và đặc biệt "
           "là ĐỀ NGHỊ CỦA VIỆN KIỂM SÁT về hướng giải quyết vụ án.")
    usr = (f"TÓM TẮT VỤ ÁN:\n{case_query}\n\n"
           f"Bên A (nguyên đơn): {a_desc}\nBên B (bị đơn): {b_desc}\n\n"
           f"Hãy tạo {n} câu truy vấn tiếng Việt ĐA DẠNG, mỗi câu nhắm tới một khía cạnh "
           f"khác nhau của vụ án để tối đa hóa số đoạn bằng chứng lấy được. "
           f'Chỉ trả về JSON: {{"queries": ["...", "..."]}}')
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


# --------------------------------------------------------------------------- #
# Stage 2 — structured case understanding + VKS stance
# --------------------------------------------------------------------------- #
def case_understanding(case_query: str, a_role: str, b_role: str,
                       a_desc: str, b_desc: str, evidence: str) -> list[dict]:
    sys = ("Bạn là thẩm phán phân tích hồ sơ vụ án dân sự Việt Nam. Đọc tóm tắt và các đoạn "
           "bằng chứng trích từ hồ sơ, rồi trích xuất thông tin có cấu trúc. "
           "Tuyệt đối KHÔNG bịa; nếu không có thông tin, để chuỗi rỗng. "
           "Đặc biệt chú ý phần Viện kiểm sát (VKS)/Kiểm sát viên đề nghị Hội đồng xét xử — "
           "đây là tín hiệu quan trọng về hướng giải quyết.")
    usr = (f"TÓM TẮT (case_query):\n{case_query}\n\n"
           f"{a_role} (A): {a_desc}\n{b_role} (B): {b_desc}\n\n"
           f"CÁC ĐOẠN BẰNG CHỨNG TỪ HỒ SƠ:\n{evidence}\n\n"
           "Trích xuất JSON đúng schema sau:\n"
           "{\n"
           '  "tom_tat": "tóm tắt tranh chấp 2-3 câu",\n'
           '  "nguyen_don_yeu_cau": "các yêu cầu chính của bên A",\n'
           '  "bi_don_phan_hoi": "phản hồi/quan điểm của bên B",\n'
           '  "tranh_chap_chinh": ["vấn đề tranh chấp 1", "..."],\n'
           '  "chung_cu_quan_trong": ["chứng cứ 1", "..."],\n'
           '  "so_tien_tai_san": "số tiền/tài sản tranh chấp nếu có",\n'
           '  "vks_de_nghi": "trích nguyên văn phần VKS đề nghị nếu có, nếu không để rỗng",\n'
           '  "vks_stance": "ACCEPT_FULL | ACCEPT_PARTIAL | REJECT | UNKNOWN",\n'
           '  "van_de_phap_ly": ["quan hệ pháp luật / vấn đề pháp lý cần áp dụng luật, ví dụ: bồi thường thiệt hại ngoài hợp đồng, tranh chấp quyền sử dụng đất, hợp đồng vay tài sản"]\n'
           "}\n"
           "vks_stance: ACCEPT_FULL nếu VKS đề nghị chấp nhận TOÀN BỘ yêu cầu nguyên đơn; "
           "ACCEPT_PARTIAL nếu chấp nhận MỘT PHẦN; REJECT nếu đề nghị BÁC yêu cầu; "
           "UNKNOWN nếu không rõ.\nChỉ trả về JSON.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


# --------------------------------------------------------------------------- #
# Stage 3 — dual-advocate debate + adjudication
# --------------------------------------------------------------------------- #
def advocate(side: str, summary: str, precedents: str) -> list[dict]:
    who = "nguyên đơn (bên A)" if side == "A" else "bị đơn (bên B)"
    goal = ("được Tòa chấp nhận yêu cầu khởi kiện" if side == "A"
            else "bác bỏ yêu cầu khởi kiện của nguyên đơn")
    sys = (f"Bạn là luật sư bảo vệ cho {who}. Hãy lập luận thuyết phục nhất, dựa trên tình "
           f"tiết và pháp luật Việt Nam, để {goal}. Ngắn gọn 3-5 gạch đầu dòng.")
    usr = f"HỒ SƠ VỤ ÁN:\n{summary}\n\nÁN LỆ THAM KHẢO:\n{precedents}\n\nLập luận của bạn:"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def adjudicator(summary: str, precedents: str, arg_a: str, arg_b: str,
                vks_hint: str) -> list[dict]:
    sys = ("Bạn là Hội đồng xét xử một Tòa án dân sự Việt Nam. Dựa trên hồ sơ, án lệ tương tự, "
           "lập luận hai bên và đề nghị của Viện kiểm sát, hãy DỰ ĐOÁN kết quả vụ án. "
           "Ở các vụ dân sự Việt Nam, Tòa thường theo sát đề nghị của Viện kiểm sát; hãy cân nhắc "
           "kỹ tín hiệu này nhưng vẫn đánh giá độc lập trên chứng cứ.\n\n" + LABEL_GLOSS)
    usr = (f"HỒ SƠ:\n{summary}\n\nÁN LỆ TƯƠNG TỰ (kèm kết quả thật):\n{precedents}\n\n"
           f"LẬP LUẬN BÊN A:\n{arg_a}\n\nLẬP LUẬN BÊN B:\n{arg_b}\n\n"
           f"ĐỀ NGHỊ CỦA VIỆN KIỂM SÁT: {vks_hint}\n\n"
           "Hãy suy luận từng bước, sau đó kết luận bằng đúng MỘT nhãn ở dòng cuối cùng theo "
           "định dạng:\nKẾT LUẬN: <A_WIN|PARTIAL_A_WIN|PARTIAL_B_WIN|B_WIN>")
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def direct_predict(summary: str, precedents: str, vks_hint: str) -> list[dict]:
    sys = ("Bạn là Hội đồng xét xử dân sự Việt Nam. Dự đoán kết quả vụ án.\n" + LABEL_GLOSS)
    usr = (f"HỒ SƠ:\n{summary}\n\nÁN LỆ TƯƠNG TỰ:\n{precedents}\n\n"
           f"ĐỀ NGHỊ VIỆN KIỂM SÁT: {vks_hint}\n\n"
           "Suy luận ngắn gọn rồi kết thúc bằng:\nKẾT LUẬN: <nhãn>")
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


# --------------------------------------------------------------------------- #
# Stage 4 — law provision selection (rerank / prune to F1-optimal set)
# --------------------------------------------------------------------------- #
def law_select(summary: str, legal_issues: str, candidates_block: str) -> list[dict]:
    sys = ("Bạn là chuyên gia pháp luật Việt Nam. Từ danh sách điều luật ỨNG VIÊN, hãy chọn ra "
           "CHÍNH XÁC những điều luật mà Tòa án sẽ VIỆN DẪN làm căn cứ giải quyết vụ án này "
           "(cả luật nội dung lẫn luật tố tụng/án phí nếu phù hợp). "
           "Ưu tiên độ chính xác: KHÔNG chọn điều không liên quan. Chỉ chọn trong danh sách ứng viên.")
    usr = (f"HỒ SƠ:\n{summary}\n\nVẤN ĐỀ PHÁP LÝ:\n{legal_issues}\n\n"
           f"ĐIỀU LUẬT ỨNG VIÊN (định dạng: [i] law_id | Điều num | trích nội dung):\n"
           f"{candidates_block}\n\n"
           'Trả về JSON gồm chỉ số các ứng viên được chọn: {"chosen": [i, j, ...]}')
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
