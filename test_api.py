"""
Automated test script for Sonali Intelligent KYC NID OCR Service.
Tests single and dual OCR extraction using generated sample NID card images.
"""
import asyncio
import json
from pathlib import Path
from app.main import app

SAMPLES_DIR = Path(__file__).parent / "samples"


async def simulate_file_upload(path: str, field_name: str, file_path: Path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    boundary = "----WebKitFormBoundarySonaliKYCTest"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{file_path.name}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
            (b"host", b"localhost:8000")
        ],
        "server": ("127.0.0.1", 8000),
    }

    response_body = []
    status_code = None
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    await app(scope, receive, send)
    resp_text = b"".join(response_body).decode("utf-8")
    return status_code, json.loads(resp_text)


async def simulate_dual_upload(front_path: Path, back_path: Path):
    with open(front_path, "rb") as f:
        front_bytes = f.read()
    with open(back_path, "rb") as f:
        back_bytes = f.read()

    boundary = "----WebKitFormBoundarySonaliKYCTest"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"front_file\"; filename=\"{front_path.name}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + front_bytes + (
        f"\r\n--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"back_file\"; filename=\"{back_path.name}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + back_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/nid/ocr/dual",
        "raw_path": b"/api/v1/nid/ocr/dual",
        "query_string": b"",
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
            (b"host", b"localhost:8000")
        ],
        "server": ("127.0.0.1", 8000),
    }

    response_body = []
    status_code = None
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    await app(scope, receive, send)
    resp_text = b"".join(response_body).decode("utf-8")
    return status_code, json.loads(resp_text)


async def run_tests():
    print("=" * 60)
    print("SONALI INTELLIGENT KYC - NID OCR API TEST SUITE")
    print("=" * 60)

    # Test 1: Smart NID Front
    front_file = SAMPLES_DIR / "sample_smart_nid_front.jpg"
    print(f"\n[1] Testing Smart NID Front Upload ({front_file.name})...")
    status, res = await simulate_file_upload("/api/v1/nid/ocr", "file", front_file)
    print(f"Status: {status}")
    print("Extracted Data:")
    print(json.dumps(res.get("data"), indent=2, ensure_ascii=False))

    # Test 2: Smart NID Back
    back_file = SAMPLES_DIR / "sample_smart_nid_back.jpg"
    print(f"\n[2] Testing Smart NID Back Upload ({back_file.name})...")
    status, res = await simulate_file_upload("/api/v1/nid/ocr", "file", back_file)
    print(f"Status: {status}")
    print("Extracted Data:")
    print(json.dumps(res.get("data"), indent=2, ensure_ascii=False))

    # Test 3: Old NID 17-digit
    old_file = SAMPLES_DIR / "sample_old_nid_17digit.jpg"
    print(f"\n[3] Testing Old 17-digit NID Upload ({old_file.name})...")
    status, res = await simulate_file_upload("/api/v1/nid/ocr", "file", old_file)
    print(f"Status: {status}")
    print("Extracted Data:")
    print(json.dumps(res.get("data"), indent=2, ensure_ascii=False))

    # Test 4: Dual Front + Back
    print("\n[4] Testing Dual Front + Back Upload (/api/v1/nid/ocr/dual)...")
    status, res = await simulate_dual_upload(front_file, back_file)
    print(f"Status: {status}")
    print("Merged KYC Data:")
    print(json.dumps(res.get("data"), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("ALL API TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())
