# =============================================================================
# PIPELINE HOÀN CHỈNH: LÀM SẠCH → TẠO ĐẶC TRƯNG → HUẤN LUYỆN MÔ HÌNH
# =============================================================================

import os
import sys
import subprocess
import time
import re
import warnings

import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from catboost import CatBoostClassifier
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# Cố định seed để tái lập kết quả
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# PHẦN 1: LÀM SẠCH DỮ LIỆU
# =============================================================================

# Từ điển teencode mở rộng (50+ mẫu)
tu_dien_teencode = {
    # Mẫu gốc
    'mik': 'mình', 'mk': 'mình', 'mjk': 'mình', 'mjnh': 'mình',
    'ik': 'đi',
    'hok': 'không', 'ko': 'không', 'k': 'không', 'hem': 'không',
    'dc': 'được', 'đc': 'được', 'duoc': 'được',
    'bit': 'biết', 'bik': 'biết', 'bt': 'biết',
    'j': 'gì', 'gj': 'gì', 'ji': 'gì',
    'vs': 'với', 'voi': 'với', 'voii': 'với',
    'h': 'giờ', 'r': 'rồi', 'roj': 'rồi', 'ròi': 'rồi',

    # Mẫu mở rộng
    'thik': 'thích', 'thjk': 'thích', 'thix': 'thích',
    'cx': 'cũng', 'cug': 'cũng', 'cugx': 'cũng',
    'nhiu': 'nhiều', 'nhju': 'nhiều', 'nhìu': 'nhiều',
    'lm': 'làm', 'lam': 'làm', 'lamf': 'làm',
    'nv': 'như vậy', 'ntn': 'như thế nào',
    'trog': 'trong', 'trongg': 'trong',
    'ng': 'người', 'ngj': 'người', 'nguoi': 'người',
    'hjc': 'hiểu', 'hju': 'hiểu', 'hjk': 'hiểu',
    'thj': 'thì', 'thi': 'thì', 'thif': 'thì',
    'nx': 'nữa', 'nưa': 'nữa', 'nửa': 'nữa',
    'bh': 'bao giờ', 'bjo': 'bao giờ',
    'ms': 'mới', 'moi': 'mới', 'mjoi': 'mới',
    'chs': 'chưa', 'ch': 'chưa', 'chwa': 'chưa',
    'nhma': 'nhưng mà', 'nma': 'nhưng mà',
    'fai': 'phải', 'pải': 'phải',
    'đag': 'đang', 'dang': 'đang', 'dangf': 'đang',
    'vx': 'vậy', 'z': 'vậy', 'zậy': 'vậy',

    # Từ dành riêng cho sinh viên
    'sv': 'sinh viên', 'gv': 'giáo viên', 'hs': 'học sinh',
    'th': 'thực hành', 'lt': 'lý thuyết',
    'đh': 'đại học', 'cd': 'cao đẳng',
    'mon': 'môn', 'hoc': 'học', 'ky': 'kỳ',
    'thi': 'thi', 'kt': 'kiểm tra',

    # Từ cảm xúc
    'tot': 'tốt', 'totj': 'tốt', 'good': 'tốt',
    'xau': 'xấu', 'te': 'tệ', 'bad': 'xấu',
    'kho': 'khó', 'khoq': 'khó',
    'de': 'dễ', 'dex': 'dễ', 'easy': 'dễ',
    'muon': 'muộn', 'late': 'muộn',
    'som': 'sớm', 'early': 'sớm',
    'vang': 'vắng', 'absent': 'vắng',
    'qá': 'quá', 'wa': 'quá', 'wá': 'quá',
    'chac': 'chắc', 'chak': 'chắc',
}


def lam_sach_van_ban_tieng_viet(van_ban):
    """Làm sạch văn bản tiếng Việt với regex và chuẩn hóa nâng cao"""
    if pd.isna(van_ban) or van_ban == "":
        return ""

    van_ban = str(van_ban).lower()

    # Sửa lỗi đánh máy phổ biến TRƯỚC khi xử lý teencode
    van_ban = van_ban.replace('tuyểển', 'tuyển')
    van_ban = van_ban.replace('hjện', 'hiện')
    van_ban = van_ban.replace('hjọc', 'học')

    # Thay thế teencode theo ranh giới từ
    for ma, thuc in tu_dien_teencode.items():
        mau = r'\b' + re.escape(ma) + r'\b'
        van_ban = re.sub(mau, thuc, van_ban)

    # Giữ lại ký tự tiếng Việt + số + khoảng trắng
    van_ban = re.sub(
        r'[^\w\s\dáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]',
        ' ', van_ban
    )

    # Xóa khoảng trắng thừa
    van_ban = re.sub(r'\s+', ' ', van_ban).strip()

    return van_ban


def lam_sach_du_lieu():
    """PHẦN 1: Làm sạch toàn bộ dữ liệu thô và xuất file đã làm sạch"""
    print("=" * 80)
    print("PHẦN 1: LÀM SẠCH DỮ LIỆU NÂNG CAO")
    print("=" * 80)

    # --- Tải dữ liệu ---
    print("\n[1/7] Đang tải dữ liệu...")
    du_lieu_train = pd.read_csv('data/train.csv')
    du_lieu_test = pd.read_csv('data/test.csv')

    du_lieu_train['dataset_source'] = 'train'
    du_lieu_test['dataset_source'] = 'test'
    du_lieu_tong = pd.concat([du_lieu_train, du_lieu_test], axis=0, ignore_index=True)

    print(f"Train: {du_lieu_train.shape}, Test: {du_lieu_test.shape}")

    # --- Làm sạch cột phân loại ---
    print("\n[2/7] Làm sạch dữ liệu phân loại...")

    # Admission_Mode
    du_lieu_tong['Admission_Mode'] = du_lieu_tong['Admission_Mode'].apply(lam_sach_van_ban_tieng_viet)

    # English_Level – ánh xạ nâng cao
    du_lieu_tong['English_Level_Clean'] = (
        du_lieu_tong['English_Level'].astype(str).str.lower().str.strip().str.rstrip('.')
    )

    bang_anh_ngu = {
        'a0': 0, 'a1': 1, 'a2': 2,
        'b1': 3, 'b2': 4,
        'c1': 5, 'c2': 6,
        'ielts 6.0+': 5, 'ielts 60+': 5, 'ielts 6+': 5,
        'ielts 7.0+': 6, 'ielts 7+': 6,
    }

    du_lieu_tong['English_Level_Mapped'] = du_lieu_tong['English_Level_Clean'].map(bang_anh_ngu)
    gia_tri_mode = du_lieu_tong['English_Level_Mapped'].mode()[0]
    du_lieu_tong['English_Level_Mapped'] = du_lieu_tong['English_Level_Mapped'].fillna(gia_tri_mode).astype(int)

    # --- Xử lý cột văn bản ---
    print("\n[3/7] Xử lý dữ liệu văn bản...")

    cot_van_ban = ['Personal_Essay', 'Advisor_Notes']

    tu_tich_cuc = ['tốt', 'giỏi', 'xuất sắc', 'tuyệt', 'thích', 'yêu', 'vui']
    tu_tieu_cuc = ['không', 'xấu', 'tệ', 'kém', 'bỏ', 'chán', 'khó', 'muộn', 'vắng']

    for cot in cot_van_ban:
        # Đặc trưng độ dài văn bản (trước khi làm sạch)
        du_lieu_tong[f'{cot}_length'] = du_lieu_tong[cot].fillna('').astype(str).str.len()
        du_lieu_tong[f'{cot}_word_count'] = du_lieu_tong[cot].fillna('').astype(str).str.split().str.len()

        # Làm sạch văn bản
        du_lieu_tong[cot] = du_lieu_tong[cot].apply(lam_sach_van_ban_tieng_viet)

        # Đặc trưng cảm xúc
        du_lieu_tong[f'{cot}_positive_count'] = du_lieu_tong[cot].fillna('').apply(
            lambda x: sum(tu in x for tu in tu_tich_cuc)
        )
        du_lieu_tong[f'{cot}_negative_count'] = du_lieu_tong[cot].fillna('').apply(
            lambda x: sum(tu in x for tu in tu_tieu_cuc)
        )
        du_lieu_tong[f'{cot}_sentiment_ratio'] = (
            du_lieu_tong[f'{cot}_positive_count'] / (du_lieu_tong[f'{cot}_negative_count'] + 1)
        )

    print("  ✓ Đã tạo đặc trưng độ dài văn bản & cảm xúc")

    # --- Làm sạch dữ liệu số ---
    print("\n[4/7] Làm sạch dữ liệu số...")

    # Tuition_Debt
    du_lieu_tong['Tuition_Debt'] = pd.to_numeric(du_lieu_tong['Tuition_Debt'], errors='coerce').fillna(0)

    # Điểm danh – xử lý ngoại lệ tích cực hơn
    cot_diem_danh = [c for c in du_lieu_tong.columns if 'Att_Subject_' in c]

    for cot in cot_diem_danh:
        du_lieu_tong[cot] = pd.to_numeric(du_lieu_tong[cot], errors='coerce')
        du_lieu_tong.loc[du_lieu_tong[cot] < 0, cot] = np.nan      # Âm → NaN
        du_lieu_tong.loc[du_lieu_tong[cot] > 20, cot] = np.nan     # > 20 → NaN
        du_lieu_tong.loc[du_lieu_tong[cot] == 0, cot] = 0.5        # 0 → 0.5

    # Count_F
    du_lieu_tong['Count_F'] = pd.to_numeric(du_lieu_tong['Count_F'], errors='coerce').fillna(0)

    # --- Mã hóa biến phân loại ---
    print("\n[5/7] Mã hóa biến phân loại...")

    cac_cot_phan_loai = ['Gender', 'Hometown', 'Current_Address', 'Club_Member', 'Admission_Mode']
    bo_ma_hoa = LabelEncoder()

    for cot in cac_cot_phan_loai:
        if cot in du_lieu_tong.columns:
            du_lieu_tong[cot] = du_lieu_tong[cot].astype(str)
            du_lieu_tong[f'{cot}_Encoded'] = bo_ma_hoa.fit_transform(du_lieu_tong[cot])

    # --- Xuất dữ liệu đã làm sạch ---
    print("\n[6/7] Xuất dữ liệu đã làm sạch...")

    train_sach = du_lieu_tong[du_lieu_tong['dataset_source'] == 'train'].drop(columns=['dataset_source'])
    test_sach = du_lieu_tong[du_lieu_tong['dataset_source'] == 'test'].drop(
        columns=['dataset_source', 'Academic_Status']
    )

    train_sach.to_csv('data/train_sach.csv', index=False)
    test_sach.to_csv('data/test_sach.csv', index=False)

    print("✅ Làm sạch dữ liệu hoàn tất!")
    print(f"   - Teencode mở rộng: {len(tu_dien_teencode)} mẫu")
    print(
        f"   - Đặc trưng văn bản: "
        f"{len([c for c in du_lieu_tong.columns if 'length' in c or 'count' in c or 'sentiment' in c])} "
        f"đặc trưng mới"
    )
    print("   - File đã lưu: train_sach.csv, test_sach.csv")


# =============================================================================
# PHẦN 2: TẠO ĐẶC TRƯNG
# =============================================================================

def lay_embedding_bert(danh_sach_van_ban, mo_hinh_bert, bo_token_hoa, thiet_bi, kich_co_lo=32):
    """Trích xuất embedding BERT với thanh tiến trình"""
    mo_hinh_bert.eval()
    danh_sach_embedding = []

    # Tiền xử lý văn bản
    print("    → Đang token hóa...")
    van_ban_da_xu_ly = [
        word_tokenize(str(t) if pd.notna(t) else "", format="text")
        for t in danh_sach_van_ban
    ]

    # Trích xuất embedding
    print("    → Đang trích xuất embedding...")
    with torch.no_grad():
        for i in tqdm(range(0, len(van_ban_da_xu_ly), kich_co_lo), desc="    "):
            lo_van_ban = van_ban_da_xu_ly[i: i + kich_co_lo]

            dau_vao = bo_token_hoa(
                lo_van_ban,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(thiet_bi)

            dau_ra = mo_hinh_bert(**dau_vao)
            embedding_cls = dau_ra.last_hidden_state[:, 0, :].cpu().numpy()
            danh_sach_embedding.append(embedding_cls)

    return np.vstack(danh_sach_embedding)


def tao_dac_trung():
    """PHẦN 2: Tạo đặc trưng nâng cao từ dữ liệu đã làm sạch"""
    print("=" * 80)
    print("PHẦN 2: TẠO ĐẶC TRƯNG NÂNG CAO")
    print("=" * 80)

    # Cấu hình thiết bị
    thiet_bi = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nSử dụng thiết bị: {thiet_bi}")

    # --- Tải dữ liệu đã làm sạch ---
    print("\n[1/5] Đang tải dữ liệu đã làm sạch...")
    du_lieu_train = pd.read_csv('data/train_sach.csv')
    du_lieu_test = pd.read_csv('data/test_sach.csv')

    du_lieu_train['dataset_source'] = 'train'
    du_lieu_test['dataset_source'] = 'test'
    du_lieu_tong = pd.concat([du_lieu_train, du_lieu_test], axis=0, ignore_index=True)

    # Ép kiểu số
    cot_diem_danh = [c for c in du_lieu_tong.columns if 'Att_Subject_' in c]
    for cot in cot_diem_danh + ['Tuition_Debt', 'Count_F']:
        du_lieu_tong[cot] = pd.to_numeric(du_lieu_tong[cot], errors='coerce')

    # --- Đặc trưng điểm danh nâng cao (20+ đặc trưng) ---
    print("\n[2/5] Tạo đặc trưng điểm danh nâng cao...")

    du_lieu_tong['Mean_Score_All'] = du_lieu_tong[cot_diem_danh].mean(axis=1)
    du_lieu_tong['Std_Score_All'] = du_lieu_tong[cot_diem_danh].std(axis=1).fillna(0)
    du_lieu_tong['Min_Score'] = du_lieu_tong[cot_diem_danh].min(axis=1)
    du_lieu_tong['Max_Score'] = du_lieu_tong[cot_diem_danh].max(axis=1)
    du_lieu_tong['Median_Score'] = du_lieu_tong[cot_diem_danh].median(axis=1)

    # Đếm các mức
    du_lieu_tong['Count_Attended'] = du_lieu_tong[cot_diem_danh].notna().sum(axis=1)
    du_lieu_tong['Count_Missing'] = du_lieu_tong[cot_diem_danh].isna().sum(axis=1)
    du_lieu_tong['Subject_Fail_Count'] = (du_lieu_tong[cot_diem_danh] < 4.0).sum(axis=1)
    du_lieu_tong['Subject_Pass_Count'] = (du_lieu_tong[cot_diem_danh] >= 4.0).sum(axis=1)

    # Mức hiệu suất chi tiết hơn
    du_lieu_tong['Subject_Excellent_Count'] = (du_lieu_tong[cot_diem_danh] >= 15.0).sum(axis=1)
    du_lieu_tong['Subject_Good_Count'] = (
        (du_lieu_tong[cot_diem_danh] >= 10.0) & (du_lieu_tong[cot_diem_danh] < 15.0)
    ).sum(axis=1)
    du_lieu_tong['Subject_Average_Count'] = (
        (du_lieu_tong[cot_diem_danh] >= 7.0) & (du_lieu_tong[cot_diem_danh] < 10.0)
    ).sum(axis=1)
    du_lieu_tong['Subject_Poor_Count'] = (
        (du_lieu_tong[cot_diem_danh] >= 4.0) & (du_lieu_tong[cot_diem_danh] < 7.0)
    ).sum(axis=1)

    # Tỷ lệ
    du_lieu_tong['Excellent_Ratio'] = du_lieu_tong['Subject_Excellent_Count'] / (du_lieu_tong['Count_Attended'] + 1)
    du_lieu_tong['Fail_Ratio'] = du_lieu_tong['Subject_Fail_Count'] / (du_lieu_tong['Count_Attended'] + 1)
    du_lieu_tong['Pass_Ratio'] = du_lieu_tong['Subject_Pass_Count'] / (du_lieu_tong['Count_Attended'] + 1)

    # Biến động
    du_lieu_tong['Score_Range'] = du_lieu_tong['Max_Score'] - du_lieu_tong['Min_Score']
    du_lieu_tong['Score_CV'] = du_lieu_tong['Std_Score_All'] / (du_lieu_tong['Mean_Score_All'] + 1)

    # Phân tích xu hướng (nửa đầu vs nửa sau)
    n_nua = len(cot_diem_danh) // 2
    cot_nua_dau = cot_diem_danh[:n_nua]
    cot_nua_sau = cot_diem_danh[n_nua:]

    du_lieu_tong['First_Half_Mean'] = du_lieu_tong[cot_nua_dau].mean(axis=1)
    du_lieu_tong['Second_Half_Mean'] = du_lieu_tong[cot_nua_sau].mean(axis=1)
    du_lieu_tong['Score_Trend'] = du_lieu_tong['Second_Half_Mean'] - du_lieu_tong['First_Half_Mean']
    du_lieu_tong['Trend_Positive'] = (du_lieu_tong['Score_Trend'] > 0).astype(int)

    so_dac_trung_diem_danh = len([
        c for c in du_lieu_tong.columns
        if 'Score' in c or 'Count' in c or 'Ratio' in c or 'Trend' in c
    ])
    print(f"  ✓ Đã tạo {so_dac_trung_diem_danh} đặc trưng điểm danh")

    # --- Đặc trưng tài chính & tương tác ---
    print("\n[3/5] Tạo đặc trưng tương tác nâng cao...")

    du_lieu_tong['Tuition_Debt'] = du_lieu_tong['Tuition_Debt'].fillna(0)
    du_lieu_tong['Has_Debt'] = (du_lieu_tong['Tuition_Debt'] > 0).astype(int)
    du_lieu_tong['Log_Debt'] = np.log1p(du_lieu_tong['Tuition_Debt'])

    # Mức độ nợ
    du_lieu_tong['Debt_Level'] = pd.cut(
        du_lieu_tong['Tuition_Debt'],
        bins=[-1, 0, 1000000, 5000000, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(int)

    du_lieu_tong['Count_F'] = du_lieu_tong['Count_F'].fillna(0)

    # Đặc trưng tương tác
    du_lieu_tong['Fail_Debt_Interaction'] = du_lieu_tong['Count_F'] * du_lieu_tong['Log_Debt']
    du_lieu_tong['Performance_Debt_Ratio'] = du_lieu_tong['Mean_Score_All'] / (du_lieu_tong['Log_Debt'] + 1)
    du_lieu_tong['Attendance_Debt_Risk'] = du_lieu_tong['Fail_Ratio'] * du_lieu_tong['Has_Debt']

    # Điểm rủi ro học tập
    du_lieu_tong['Academic_Risk_Score'] = (
        du_lieu_tong['Fail_Ratio'] * 0.4 +
        (1 - du_lieu_tong['Pass_Ratio']) * 0.3 +
        du_lieu_tong['Count_F'] / 10 * 0.2 +
        du_lieu_tong['Has_Debt'] * 0.1
    )

    so_dac_trung_tuong_tac = len([
        c for c in du_lieu_tong.columns
        if 'Interaction' in c or 'Risk' in c or 'Level' in c
    ])
    print(f"  ✓ Đã tạo {so_dac_trung_tuong_tac} đặc trưng tương tác")

    # --- Embedding PhoBERT cho cả hai cột văn bản ---
    # QUAN TRỌNG: Luôn chạy PhoBERT, không skip theo điều kiện GPU/size
    print("\n[4/5] Xử lý embedding PhoBERT...")
    print(f"  Thiết bị: {thiet_bi} | Số mẫu: {len(du_lieu_tong)}")
    print("  (Có thể mất 5-10 phút tùy GPU, CPU sẽ lâu hơn nhưng vẫn chạy)")

    ten_mo_hinh = "vinai/phobert-base"
    bo_token_hoa = AutoTokenizer.from_pretrained(ten_mo_hinh, use_fast=False)
    mo_hinh_bert = AutoModel.from_pretrained(ten_mo_hinh).to(thiet_bi)

    # Xử lý Advisor_Notes
    print("\n  [A] Xử lý Advisor_Notes...")
    van_ban_ghi_chu = du_lieu_tong['Advisor_Notes'].fillna("").tolist()
    embedding_ghi_chu = lay_embedding_bert(van_ban_ghi_chu, mo_hinh_bert, bo_token_hoa, thiet_bi)

    print("    → Giảm chiều (PCA)...")
    pca_ghi_chu = PCA(n_components=16, random_state=42)
    ghi_chu_pca = pca_ghi_chu.fit_transform(embedding_ghi_chu)

    for i in range(16):
        du_lieu_tong[f'BERT_Note_{i}'] = ghi_chu_pca[:, i]

    print(f"    ✓ Phương sai giải thích: {pca_ghi_chu.explained_variance_ratio_.sum():.2%}")

    # Xử lý Personal_Essay
    print("\n  [B] Xử lý Personal_Essay...")
    van_ban_essay = du_lieu_tong['Personal_Essay'].fillna("").tolist()
    embedding_essay = lay_embedding_bert(van_ban_essay, mo_hinh_bert, bo_token_hoa, thiet_bi)

    print("    → Giảm chiều (PCA)...")
    pca_essay = PCA(n_components=16, random_state=42)
    essay_pca = pca_essay.fit_transform(embedding_essay)

    for i in range(16):
        du_lieu_tong[f'BERT_Essay_{i}'] = essay_pca[:, i]

    print(f"    ✓ Phương sai giải thích: {pca_essay.explained_variance_ratio_.sum():.2%}")

    # Giải phóng bộ nhớ GPU sau khi xong BERT
    del mo_hinh_bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  ✓ Đã giải phóng bộ nhớ GPU")

    # --- Đặc trưng đa thức ---
    print("\n[5/5] Tạo đặc trưng đa thức...")

    cac_dac_trung_chinh = ['Mean_Score_All', 'Fail_Ratio', 'Log_Debt']

    for dac_trung in cac_dac_trung_chinh:
        if dac_trung in du_lieu_tong.columns:
            du_lieu_tong[f'{dac_trung}_squared'] = du_lieu_tong[dac_trung] ** 2
            du_lieu_tong[f'{dac_trung}_cubed'] = du_lieu_tong[dac_trung] ** 3

    so_dac_trung_da_thuc = len([c for c in du_lieu_tong.columns if 'squared' in c or 'cubed' in c])
    print(f"  ✓ Đã tạo {so_dac_trung_da_thuc} đặc trưng đa thức")

    # --- Xuất đặc trưng cuối ---
    print("\n[6/6] Xuất đặc trưng nâng cao...")

    train_dac_trung = du_lieu_tong[du_lieu_tong['dataset_source'] == 'train'].drop(columns=['dataset_source'])
    test_dac_trung = du_lieu_tong[du_lieu_tong['dataset_source'] == 'test'].drop(
        columns=['dataset_source', 'Academic_Status']
    )

    train_dac_trung.to_csv('data/train_dac_trung.csv', index=False)
    test_dac_trung.to_csv('data/test_dac_trung.csv', index=False)

    tong_so_dac_trung = len(train_dac_trung.columns) - 1
    print(f"\n{'=' * 80}")
    print("TỔNG KẾT TẠO ĐẶC TRƯNG")
    print(f"{'=' * 80}")
    print(f"Tổng số đặc trưng: {tong_so_dac_trung}")
    print("  - Đặc trưng điểm danh: ~25")
    print("  - Độ dài văn bản / cảm xúc: ~12")
    print("  - Embedding PhoBERT: 32 (16 mỗi cột)")
    print("  - Đặc trưng tương tác: ~10")
    print("  - Đặc trưng đa thức: ~9")
    print(f"{'=' * 80}")
    print("✅ Tạo đặc trưng nâng cao hoàn tất!")
    print("   File đã lưu: train_dac_trung.csv, test_dac_trung.csv")


# =============================================================================
# PHẦN 3: HUẤN LUYỆN MÔ HÌNH VÀ DỰ ĐOÁN
# =============================================================================

def them_dac_trung_tac_dong(du_lieu):
    """Thêm đặc trưng tác động cho XGBoost"""
    du_lieu = du_lieu.copy()
    if 'Has_Debt' in du_lieu.columns and 'Mean_Score_All' in du_lieu.columns:
        du_lieu['debt_score_ratio'] = du_lieu['Has_Debt'] * (10 - du_lieu['Mean_Score_All'])
    if 'Personal_Essay' in du_lieu.columns:
        du_lieu['essay_len'] = du_lieu['Personal_Essay'].str.len().fillna(0)
    return du_lieu


def train_model():
    """PHẦN 3: Huấn luyện ensemble mô hình và tạo file dự đoán cuối"""
    print("=" * 80)
    print("PHẦN 3: HUẤN LUYỆN MÔ HÌNH VÀ DỰ ĐOÁN")
    print("=" * 80)

    # Reset seed trước khi train để tránh bị nhiễm từ PhoBERT / các bước trước
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print("  ✓ Đã reset seed (42) trước khi huấn luyện")

    # Tải dữ liệu đặc trưng
    du_lieu_train = pd.read_csv('data/train_dac_trung.csv')
    du_lieu_test = pd.read_csv('data/test_dac_trung.csv')

    # =========================================================================
    # A. HUẤN LUYỆN CATBOOST (70% trọng số)
    # =========================================================================
    print("\n[1/4] Đang huấn luyện CatBoost (70% trọng số)...")

    X_so = du_lieu_train.select_dtypes(include=['number']).drop(
        columns=['Academic_Status', 'Student_ID'], errors='ignore'
    )
    y_nhan = du_lieu_train['Academic_Status'].astype(int)
    X_test_so = du_lieu_test.select_dtypes(include=['number']).drop(
        columns=['Student_ID'], errors='ignore'
    )[X_so.columns]

    trong_so_lop_cat = [1.0, 1.8, 2.5]
    phan_chia_cat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xac_suat_test_cat = np.zeros((len(du_lieu_test), 3))

    for chi_so_train, chi_so_val in phan_chia_cat.split(X_so, y_nhan):
        X_tr, X_val = X_so.iloc[chi_so_train], X_so.iloc[chi_so_val]
        y_tr, y_val = y_nhan.iloc[chi_so_train], y_nhan.iloc[chi_so_val]

        mo_hinh_cat = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=7,
            class_weights=trong_so_lop_cat, loss_function='MultiClass',
            eval_metric='TotalF1', task_type="GPU", verbose=0,
            early_stopping_rounds=100,
            random_state=10
        )
        mo_hinh_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        xac_suat_test_cat += mo_hinh_cat.predict_proba(X_test_so) / 5

    # =========================================================================
    # B. HUẤN LUYỆN XGBOOST (30% trọng số)
    # =========================================================================
    print("\n[2/4] Đang huấn luyện XGBoost với đặc trưng tác động (30% trọng số)...")

    train_xgb = them_dac_trung_tac_dong(du_lieu_train)
    test_xgb = them_dac_trung_tac_dong(du_lieu_test)

    X_xgb = train_xgb.select_dtypes(include=['number']).drop(
        columns=['Academic_Status', 'Student_ID'], errors='ignore'
    )
    y_xgb = train_xgb['Academic_Status'].astype(int)
    X_test_xgb = test_xgb.select_dtypes(include=['number']).drop(
        columns=['Student_ID'], errors='ignore'
    )[X_xgb.columns]

    mo_hinh_xgb = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.015, max_depth=5,
        subsample=0.7, colsample_bytree=0.7, tree_method='hist', device='cuda',
        random_state=42
    )
    mo_hinh_xgb.fit(X_xgb, y_xgb)
    xac_suat_test_xgb = mo_hinh_xgb.predict_proba(X_test_xgb)

    # =========================================================================
    # C. TRỘN HAI MÔ HÌNH → BẢN BASELINE
    # =========================================================================
    trong_so_xgb = 0.30
    trong_so_cat = 0.70
    he_so_dropout = 1.15

    xac_suat_nen_tang = (xac_suat_test_xgb * trong_so_xgb) + (xac_suat_test_cat * trong_so_cat)
    xac_suat_nen_tang[:, 2] *= he_so_dropout
    du_doan_nen_tang = np.argmax(xac_suat_nen_tang, axis=1)

    du_lieu_nen_tang = pd.DataFrame({
        'Student_ID': du_lieu_test['Student_ID'],
        'Academic_Status': du_doan_nen_tang
    })

    # =========================================================================
    # D. ENSEMBLE ELITE VÀ LỌC THEO XÁC SUẤT
    # =========================================================================
    print("\n[3/4] Chạy Ensemble 4-Fold (Lọc độ tin cậy ELITE)...")

    # Chuẩn bị dữ liệu đầy đủ (bao gồm cả cột phân loại)
    X_day_du = du_lieu_train.drop(columns=['Academic_Status', 'Student_ID'], errors='ignore')
    y_day_du = du_lieu_train['Academic_Status'].astype(int)
    X_test_day_du = du_lieu_test[X_day_du.columns].copy()

    cac_cot_phan_loai = X_day_du.select_dtypes(include=['object', 'category']).columns.tolist()
    for cot in cac_cot_phan_loai:
        X_day_du[cot] = X_day_du[cot].astype(str).replace(['nan', 'NaN', 'None'], 'Unknown')
        X_test_day_du[cot] = X_test_day_du[cot].astype(str).replace(['nan', 'NaN', 'None'], 'Unknown')

    cac_cot_so = X_day_du.select_dtypes(include=['number']).columns.tolist()
    X_day_du[cac_cot_so] = X_day_du[cac_cot_so].fillna(0)
    X_test_day_du[cac_cot_so] = X_test_day_du[cac_cot_so].fillna(0)

    phan_chia_elite = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    xac_suat_test_elite = np.zeros((len(X_test_day_du), 3))
    trong_so_elite = {0: 0.9, 1: 1.8, 2: 3.8}

    for so_fold, (chi_so_train, _) in enumerate(phan_chia_elite.split(X_day_du, y_day_du)):
        X_tr, y_tr = X_day_du.iloc[chi_so_train], y_day_du.iloc[chi_so_train]

        mo_hinh_elite = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=6,
            class_weights=trong_so_elite, cat_features=cac_cot_phan_loai,
            task_type="GPU", verbose=0
        )
        mo_hinh_elite.fit(X_tr, y_tr)
        xac_suat_test_elite += mo_hinh_elite.predict_proba(X_test_day_du) / 5
        print(f"  ✓ Đã xong Fold {so_fold + 1}/4")

    print("\n[4/4] Kích hoạt LỌC ELITE (Ngưỡng > 80%)...")
    du_lieu_elite = du_lieu_nen_tang.copy()
    nguong_tin_cay = 0.80
    so_thay_doi = 0

    for i in range(len(du_lieu_elite)):
        xac_suat_moi = xac_suat_test_elite[i]
        nhan_moi = np.argmax(xac_suat_moi)
        do_tin_cay = np.max(xac_suat_moi)
        nhan_cu = du_lieu_nen_tang.at[i, 'Academic_Status']

        if nhan_cu != nhan_moi and do_tin_cay > nguong_tin_cay:
            du_lieu_elite.at[i, 'Academic_Status'] = nhan_moi
            so_thay_doi += 1

    # Xuất file kết quả cuối
    ten_file_xuat = 'submission_FINAL.csv'
    du_lieu_elite.to_csv(ten_file_xuat, index=False)

    print("\n" + "=" * 80)
    print("HOÀN THÀNH PIPELINE TỪ A-Z!")
    print(f"Số lượng thay đổi (Elite corrections): {so_thay_doi}")
    print(f"Đã xuất file: {ten_file_xuat}")
    print(f"Phân bổ nhãn cuối: {du_lieu_elite['Academic_Status'].value_counts().to_dict()}")
    print("=" * 80)


# =============================================================================
# HÀM MAIN: CHẠY TOÀN BỘ PIPELINE
# =============================================================================

def main():
    """Chạy toàn bộ pipeline: Làm sạch → Tạo đặc trưng → Huấn luyện mô hình"""
    print("\n" + "=" * 80)
    print(" BẮT ĐẦU PIPELINE PHÂN TÍCH HỌC THUẬT")
    print("=" * 80)

    co_file_fe = (
        os.path.exists('data/train_dac_trung.csv') and
        os.path.exists('data/test_dac_trung.csv')
    )

    if co_file_fe:
        print("\n✅ Phát hiện file feature có sẵn (train_dac_trung.csv / test_dac_trung.csv)")
        print("   → Bỏ qua bước Làm sạch & Tạo đặc trưng, dùng lại file đã có")
        print("   → Nếu muốn tạo lại từ đầu, xóa 2 file trên rồi chạy lại\n")
    else:
        # Bước 1: Làm sạch dữ liệu
        thoi_gian_bat_dau = time.time()
        lam_sach_du_lieu()
        print(f"\n⏱ Thời gian làm sạch: {time.time() - thoi_gian_bat_dau:.1f}s\n")

        # Bước 2: Tạo đặc trưng
        thoi_gian_bat_dau = time.time()
        tao_dac_trung()
        print(f"\n⏱ Thời gian tạo đặc trưng: {time.time() - thoi_gian_bat_dau:.1f}s\n")

    # Bước 3: Huấn luyện mô hình (luôn chạy)
    thoi_gian_bat_dau = time.time()
    train_model()
    print(f"\n⏱ Thời gian huấn luyện: {time.time() - thoi_gian_bat_dau:.1f}s\n")

    print("\n🎉 PIPELINE HOÀN CHỈNH - KẾT QUẢ ĐÃ ĐƯỢC LƯU VÀO submission_FINAL.csv")


if __name__ == "__main__":
    main()