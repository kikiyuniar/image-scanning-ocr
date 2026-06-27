from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from paddleocr import PaddleOCR
from PIL import Image, ImageOps

from pillow_heif import register_heif_opener
from dotenv import load_dotenv

import cv2
import numpy as np
import pandas as pd  # Ditambahkan untuk handle Excel/CSV
import re
import os
import io
import time

load_dotenv()
register_heif_opener()

APP_NAME = os.getenv("APP_NAME", "Scanning OCR KK")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic")
VALID_EXCEL_EXTENSIONS = (".xlsx", ".xls", ".csv")

# Definisi header wajib untuk import data penduduk
EXPECTED_HEADERS = [
    'alamat', 'dusun', 'rw', 'rt', 'nama', 'no_kk', 'nik', 
    'sex', 'tempatlahir', 'tanggallahir', 'nama_ayah', 'nama_ibu'
]

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

app = FastAPI(
    title=APP_NAME,
    description="REST API OCR Kartu Keluarga & Import Data Penduduk",
    version=APP_VERSION
)

# ==========================================
# OCR INIT
# ==========================================
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False,
    enable_mkldnn=False
)

# ==========================================
# RESPONSE HELPER
# ==========================================
def error_response(code: str, message: str, detail=None):
    response = {
        "status": "error",
        "error_code": code,
        "message": message
    }
    if detail:
        response["detail"] = str(detail)
    return response

# ==========================================
# IMPORT EXCEL / CSV DATA PENDUDUK (NEW)
# ==========================================
@app.post("/import-penduduk")
async def import_penduduk(file: UploadFile = File(...)):
    filename = file.filename.lower()
    
    if not filename.endswith(VALID_EXCEL_EXTENSIONS):
        return error_response(
            "INVALID_FILE_FORMAT", 
            "Format file tidak didukung. Harus berupa .xlsx, .xls, atau .csv"
        )
    
    try:
        contents = await file.read()
        
        # Ambil bytes data, paksa semua kolom menjadi string sejak awal (dtype=str)
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), dtype=str)
        else:
            df = pd.read_excel(io.BytesIO(contents), dtype=str)
        
        # Bersihkan spasi pada header
        df.columns = df.columns.str.strip()
        current_headers = df.columns.tolist()
        
        # Pengecekan Header
        missing_headers = [h for h in EXPECTED_HEADERS if h not in current_headers]
        if missing_headers:
            return error_response(
                "INVALID_HEADER_STRUCTURE",
                f"Struktur kolom dokumen salah. Kolom berikut wajib ada: {', '.join(missing_headers)}"
            )
            
        # PENTING: Bersihkan data kosong (NaN/None) dan pastikan murni string standar Python
        df = df.astype(str).replace(['nan', 'NaN', 'None', '<NA>'], '')
        df = df.fillna("")
        
        total_rows = len(df)
        
        # Gunakan orient="records" namun kita pastikan lagi tipenya aman untuk json serializer fastapi
        data_records = df.to_dict(orient="records")
        
        return {
            "status": "success",
            "message": f"Berhasil memproses {total_rows} data penduduk.",
            "total": total_rows,
            "data": data_records
        }
        
    except Exception as e:
        return error_response(
            "IMPORT_FAILED",
            "Terjadi kesalahan saat mengekstrak file excel/csv",
            str(e)
        )

# ==========================================
# SCAN OCR
# ==========================================
@app.post("/scan")
async def scan_kk(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        
        # ==========================================
        # VALIDASI EKSTENSI
        # ==========================================
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.heic')
        if not filename.endswith(valid_extensions):
            return {
                "status": "error", 
                "message": f"Format file '{file.filename}' tidak didukung. Harap unggah gambar (JPG/PNG/HEIC)."
            }

        file_bytes = await file.read()

        # ==========================================
        # PROSES PEMBACAAN GAMBAR & ROTASI otomatis (EXIF)
        # ==========================================
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            # Mengatasi masalah foto miring/terbalik akibat kamera HP
            pil_img = ImageOps.exif_transpose(pil_img)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            # Jalur cadangan jika pembacaan metadata PIL gagal
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "Gambar rusak atau tidak valid"}

        # =====================
        # UPSCALE
        # =====================
        h, w = img.shape[:2]
        if w < 2000:
            scale = 2000 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # =====================
        # PREPROCESSING: GRAYSCALE
        # =====================
        # Mengubah ke hitam-putih agar background bising hilang dan teks lebih tajam
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # =====================
        # OCR (PERBAIKAN: Hapus cls=True di sini)
        # =====================
        result = ocr.ocr(gray)

        if not result or not result[0]:
            return {
                "status": "warning",
                "message": "OCR tidak menemukan teks pada gambar",
                "data": []
            }

        extracted_text = []
        full_text = []

        # =====================
        # PARSING OCR RESULT WITH SAFETY CHECK
        # =====================
        for line in result[0]:
            try:
                # 1. Pastikan objek 'line' tidak kosong
                if line is None:
                    continue
                
                # 2. Pastikan struktur line memiliki minimal 2 elemen [box, [text, score]]
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                
                # 3. Pastikan elemen kedua (indeks 1) berisi data text & score
                if line[1] is None or len(line[1]) < 2:
                    continue
                    
                text = str(line[1][0]).strip()
                score = float(line[1][1])

                if text:
                    extracted_text.append({
                        "text": text,
                        "confidence": round(score, 4)
                    })
                    full_text.append(text)
            except Exception:
                # Jika ada struktur aneh lainnya, lewati dengan aman tanpa membuat server crash
                continue

        full_text_str = "\n".join(full_text)

        # =====================
        # EXTRACT NOMOR KK
        # =====================
        nomor_kk = None
        kk_match = re.search(r'\d{16}', full_text_str)
        if kk_match:
            nomor_kk = kk_match.group()

        # =====================
        # EXTRACT NIK
        # =====================
        nik_list = list(set(re.findall(r'\d{16}', full_text_str)))

        if nomor_kk and nomor_kk in nik_list:
            nik_list.remove(nomor_kk)

        # =====================
        # RESPONSE
        # =====================
        return {
            "status": "success",
            "filename": file.filename,
            "nomor_kk": nomor_kk,
            "jumlah_nik": len(nik_list),
            "nik": nik_list,
            "data": extracted_text,
            "raw_text": full_text_str
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ==========================================
# COMPRESS IMAGE
# ==========================================
@app.post("/compress")
async def compress_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes))

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        max_width = 1800
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)), Image.LANCZOS)

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=80, optimize=True)
        output.seek(0)

        original_name = file.filename.rsplit(".", 1)[0]
        return StreamingResponse(
            output,
            media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{original_name}_compressed.jpg"'}
        )
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)