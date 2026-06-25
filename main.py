from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from paddleocr import PaddleOCR
from PIL import Image, ImageOps

from pillow_heif import register_heif_opener
from dotenv import load_dotenv

import cv2
import numpy as np
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

MAX_FILE_SIZE = int(
    os.getenv("MAX_FILE_SIZE", 10485760)
)

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".heic"
)

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

app = FastAPI(
    title=APP_NAME,
    description="REST API OCR Kartu Keluarga menggunakan PaddleOCR",
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
def error_response(
    code: str,
    message: str,
    detail=None
):
    response = {
        "status": "error",
        "error_code": code,
        "message": message
    }

    if detail:
        response["detail"] = str(detail)

    return response


# ==========================================
# SCAN OCR
# ==========================================
@app.post("/scan")
async def scan_kk(file: UploadFile = File(...)):

    total_start = time.time()

    try:

        filename = file.filename.lower()

        if not filename.endswith(VALID_EXTENSIONS):

            return error_response(
                "INVALID_FILE_FORMAT",
                "Format file tidak didukung"
            )

        file_bytes = await file.read()

        if len(file_bytes) == 0:

            return error_response(
                "EMPTY_FILE",
                "File yang diunggah kosong"
            )

        if len(file_bytes) > MAX_FILE_SIZE:

            return error_response(
                "FILE_TOO_LARGE",
                "Ukuran file maksimal 10 MB"
            )

        try:

            pil_img = Image.open(
                io.BytesIO(file_bytes)
            )

            pil_img = ImageOps.exif_transpose(
                pil_img
            )

            img = cv2.cvtColor(
                np.array(pil_img),
                cv2.COLOR_RGB2BGR
            )

        except Exception:

            nparr = np.frombuffer(
                file_bytes,
                np.uint8
            )

            img = cv2.imdecode(
                nparr,
                cv2.IMREAD_COLOR
            )

        if img is None:

            return error_response(
                "INVALID_IMAGE",
                "Gambar rusak atau tidak valid"
            )

        h, w = img.shape[:2]

        if w > 1800:

            scale = 1800 / w

            img = cv2.resize(
                img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

        elif w < 1200:

            scale = 1200 / w

            img = cv2.resize(
                img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        ocr_start = time.time()

        result = ocr.ocr(gray)

        ocr_time = round(
            time.time() - ocr_start,
            2
        )

        if not result or not result[0]:

            return {
                "status": "warning",
                "error_code": "TEXT_NOT_FOUND",
                "message": "OCR tidak menemukan teks",
                "ocr_time": ocr_time,
                "data": []
            }

        extracted_text = []
        full_text = []

        for line in result[0]:

            try:

                if line is None:
                    continue

                if len(line) < 2:
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
                continue

        raw_text = "\n".join(full_text)

        nomor_kk = None

        kk_match = re.search(
            r"\d{16}",
            raw_text
        )

        if kk_match:
            nomor_kk = kk_match.group()

        nik_list = list(
            set(
                re.findall(
                    r"\d{16}",
                    raw_text
                )
            )
        )

        if nomor_kk and nomor_kk in nik_list:
            nik_list.remove(nomor_kk)

        total_time = round(
            time.time() - total_start,
            2
        )

        print("=" * 60)
        print(f"FILE       : {file.filename}")
        print(f"OCR TIME   : {ocr_time}s")
        print(f"TOTAL TIME : {total_time}s")
        print(f"KK         : {nomor_kk}")
        print(f"JUMLAH NIK : {len(nik_list)}")
        print("=" * 60)

        return {
            "status": "success",
            "filename": file.filename,
            "nomor_kk": nomor_kk,
            "jumlah_nik": len(nik_list),
            "nik": nik_list,
            "ocr_time": ocr_time,
            "total_time": total_time,
            "data": extracted_text,
            "raw_text": raw_text
        }

    except Exception as e:

        return error_response(
            "INTERNAL_SERVER_ERROR",
            "Terjadi kesalahan saat memproses OCR",
            str(e)
        )

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

        # Resize jika terlalu besar
        max_width = 1800

        if image.width > max_width:

            ratio = max_width / image.width

            image = image.resize(
                (
                    max_width,
                    int(image.height * ratio)
                ),
                Image.LANCZOS
            )

        # Compress
        output = io.BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=80,
            optimize=True
        )

        output.seek(0)

        original_name = file.filename.rsplit(".", 1)[0]

        return StreamingResponse(
            output,
            media_type="image/jpeg",
            headers={
                "Content-Disposition":
                    f'attachment; filename="{original_name}_compressed.jpg"'
            }
        )

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True
    )