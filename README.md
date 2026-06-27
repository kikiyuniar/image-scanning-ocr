# OCR Image Scanner API

Repositori ini berisi aplikasi API berbasis **FastAPI** untuk memindai teks dari gambar menggunakan **PaddleOCR**. API ini mendukung berbagai format gambar standar serta format HEIF/HEIC.

## 🚀 Fitur Utama
* **Scan Gambar ke Teks:** Mengekstrak teks dari gambar secara akurat menggunakan PaddleOCR.
* **Performa Cepat:** Dibangun di atas FastAPI dan Uvicorn untuk penanganan *request* yang asinkron dan cepat.
* **Dukungan Format Luas:** Mendukung format gambar umum (JPEG, PNG) serta format Apple (HEIF/HEIC).
* **Pemrosesan Gambar:** Menggunakan OpenCV dan Pillow untuk penanganan *image* sebelum/sesudah diproses.

---

## 🛠️ Persyaratan Sistem (Requirements)

Proyek ini memerlukan **Python 3.8+** dan beberapa *library* utama berikut:

* `fastapi` - Framework web untuk membuat API.
* `uvicorn` - Server ASGI untuk menjalankan FastAPI.
* `python-multipart` - Dibutuhkan untuk menangani unggahan file gambar.
* `python-dotenv` - Untuk mengatur environment variables.
* `opencv-python` & `numpy` - Untuk manipulasi dan pemrosesan gambar.
* `pillow` & `pillow-heif` - Untuk membuka dan mengonversi berbagai format gambar (termasuk HEIF).
* `paddleocr==2.9.1` - Framework OCR utama.
* `paddlepaddle==2.6.2` - Deep learning engine pendukung PaddleOCR.

---

## 🔄 Alur & Langkah Proses Scanning Gambar (OCR)

Aplikasi backend ini memproses gambar melalui beberapa langkah terstruktur sebelum mengembalikan teks kepada pengguna:

### 1. Penerimaan File (Upload)
* User mengunggah gambar melalui endpoint API (`POST /scan`) menggunakan format `multipart/form-data`.
* Aplikasi membaca file tersebut dalam bentuk *binary data* menggunakan `python-multipart`.

### 2. Validasi & Dekoding Gambar
* Jika gambar berformat **HEIF/HEIC** (dari perangkat iOS), pustaka `pillow-heif` akan mendeteksi dan mengonversinya terlebih dahulu menjadi format gambar standar yang dikenali Python.
* File biner gambar kemudian dibaca oleh **Pillow (PIL)** dan dikonversi menjadi *array* **NumPy** agar bisa diolah oleh **OpenCV** (`cv2`).

### 3. Pra-pemrosesan Gambar (Image Preprocessing - Opsional/Sesuai Kebutuhan)
* Gambar diubah menjadi skala abu-abu (*grayscale*) atau disesuaikan kontrasnya menggunakan OpenCV untuk meningkatkan keterbacaan karakter teks oleh mesin OCR.

### 4. Deteksi dan Rekognisi Teks (Proses OCR)
* Array gambar yang sudah siap dikirim ke engine **PaddleOCR**.
* **PaddlePaddle** (sebagai *deep learning framework* di balik PaddleOCR) menjalankan model deteksi objek untuk mencari area yang berisi teks (*Text Detection*).
* Model kemudian mengenali karakter/kata di dalam area tersebut (*Text Recognition*).

### 5. Pengembalian Hasil (Response JSON)
* Engine OCR mengembalikan koordinat lokasi teks, teks yang terdeteksi, beserta tingkat akurasinya (*confidence score*).
* FastAPI menyusun data ini menjadi format JSON dan mengirimkannya kembali ke klien.

---

## 💻 Cara Instalasi dan Penggunaan

### 1. Clone Repositori
`git clone [https://github.com/username/nama-repo.git](https://github.com/username/nama-repo.git)
cd nama-repo`

### 2. Buat Virtual Environment
`python -m venv venv`

### 3. Aktifkan Virtual Environment
# Aktifkan di Windows:
`venv\Scripts\activate`
# Aktifkan di Linux/macOS:
`source venv/bin/activate`

### 4. Instal Dependencies
`pip install -r requirements.txt`

### 5. Jalankan Aplikasi
`uvicorn main:app --reload`

## Dokumentasi API
* Setelah server berjalan, Anda dapat mengakses dokumentasi interaktif (Swagger UI) di: 👉 http://127.0.0.1:8000/docs
