# Sonali Intelligent KYC - Bangladeshi NID OCR Service

High-performance FastAPI service designed for Bangladeshi National ID (NID) cards OCR and automated information extraction. Built with **Tesseract OCR (Bengali + English)** and **OpenCV** image preprocessing.

---

## 🚀 Features

- **Multi-Format Support**: Supports **Smart NID Cards (10 digits)**, **Old NID Cards (13 or 17 digits)**, and **Back Side** of cards.
- **Bilingual Extraction**: Extracts both **Bengali (`ben`)** and **English (`eng`)** texts.
- **Digit Normalization**: Automatically converts Bengali numerals (`০-৯`) to standard ASCII digits (`0-9`).
- **Date Standardization**: Parses varied date formats (`DD Mon YYYY`, `DD/MM/YYYY`, Bengali months) into standardized ISO `YYYY-MM-DD`.
- **Address & Back-Side Parsing**: Automatically extracts Blood Group, Place of Birth, Issue Date, and structured Address components (Holding, Village/Road, Post Office, Postal Code, Thana/Upazila, District).
- **Computer Vision Preprocessing**: CLAHE contrast enhancement, deskewing/tilt correction, noise reduction, and margin padding to prevent character clipping.
- **Multi-Pass OCR Fallback**: If single pass misses fields, automatically tries complementary enhancement passes to maximize KYC match rate.
- **Ready for React**: Fully configured with CORS middleware (`allow_origins=["*"]`) for easy frontend integration.

---

## 📁 Project Structure

```
ocr-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoints, CORS, file validation
│   ├── image_processor.py   # OpenCV preprocessing, deskewing, CLAHE, padding
│   ├── ocr_engine.py        # Tesseract wrapper with PSM optimization & multi-pass
│   ├── nid_extractor.py     # Regex & heuristics parser for Bangladeshi NID fields
│   └── requirements.txt
├── samples/                 # Ready-to-use sample NID card images for testing
│   ├── sample_smart_nid_front.jpg
│   ├── sample_smart_nid_back.jpg
│   └── sample_old_nid_17digit.jpg
├── Sonali_KYC_NID_OCR.postman_collection.json # 1-click Postman collection
├── test_api.py              # Automated test suite
└── requirements.txt
```

---

## 🛠️ Getting Started

### 1. Start the API Server

From the `ocr-service` directory:

```bash
# Activate your virtual environment
source venv/bin/activate

# Start Uvicorn dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at:
- **Base URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🧪 Testing with Postman

### Option A: Import the Collection (Recommended)
1. Open **Postman**.
2. Click **Import** in the top left.
3. Select the file: `Sonali_KYC_NID_OCR.postman_collection.json`.
4. The 4 pre-configured requests will appear in your sidebar!

### Option B: Manual Postman Setup

#### Single Image OCR (`POST http://localhost:8000/api/v1/nid/ocr`)
1. Set Method: `POST`
2. URL: `http://localhost:8000/api/v1/nid/ocr`
3. Go to **Body** tab -> select **form-data**
4. Set Key: `file` -> Change key type from *Text* to **File**
5. Select a sample image (e.g. from `samples/sample_smart_nid_front.jpg`)
6. Click **Send**!

#### Dual Image OCR (`POST http://localhost:8000/api/v1/nid/ocr/dual`)
1. Set Method: `POST`
2. URL: `http://localhost:8000/api/v1/nid/ocr/dual`
3. Go to **Body** tab -> select **form-data**
4. Key 1: `front_file` (Type: **File**) -> select front image
5. Key 2: `back_file` (Type: **File**) -> select back image
6. Click **Send**!

---

## ⚡ Testing via cURL

### Single Image:
```bash
curl -X POST "http://localhost:8000/api/v1/nid/ocr" \
  -H "Accept: application/json" \
  -F "file=@samples/sample_smart_nid_front.jpg"
```

### Dual-Side (Front + Back):
```bash
curl -X POST "http://localhost:8000/api/v1/nid/ocr/dual" \
  -H "Accept: application/json" \
  -F "front_file=@samples/sample_smart_nid_front.jpg" \
  -F "back_file=@samples/sample_smart_nid_back.jpg"
```

---

## 📋 Sample API Response

```json
{
  "success": true,
  "message": "Dual-side NID OCR completed successfully",
  "data": {
    "nid_number": "7351940284",
    "nid_type": "Smart NID (10 digits)",
    "name_en": "TANVIR AHMED",
    "name_bn": "তানভীর আহমেদ",
    "father_name": "মোহাম্মদ আলী",
    "mother_name": "ফাতেমা বেগম",
    "date_of_birth": "1995-08-14",
    "date_of_birth_display": "14 Aug 1995",
    "date_of_birth_raw": "14 Aug 1995",
    "blood_group": "B+",
    "place_of_birth": "DHAKA",
    "issue_date": "15/07/2019",
    "address": {
      "full_address": "বাসা/হোল্ডিং: ৪২/ক, গ্রাম/রাস্তা: গ্রীন রোড, ডাকঘর: নিউ মার্কেট - ১২০৫, থানা: ধানমন্ডি, জেলা: ঢাকা",
      "holding": "৪২/ক",
      "village_or_road": "গ্রীন রোড",
      "post_office": "নিউ মার্কেট - ১২০৫",
      "postal_code": "১২০৫",
      "upazila_or_thana": "ধানমন্ডি",
      "district": "ঢাকা"
    },
    "card_side": "both",
    "validation": {
      "is_valid_nid": true,
      "confidence_score": 1.0,
      "missing_critical_fields": [],
      "warnings": []
    }
  },
  "raw_text": {
    "front": "গণপ্রজাতন্ত্রী বাংলাদেশ সরকার\nNational ID Card\n...",
    "back": "ঠিকানা: বাসা/হোল্ডিং: ৪২/ক..."
  }
}
```

---

## 🧪 Run Automated Tests

To test the entire pipeline without Postman:
```bash
./venv/bin/python test_api.py
```


## RUN Locally
./venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000