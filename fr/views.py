from django.shortcuts import render
import os
import csv
import json
import base64
import binascii
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, time as dtime, timedelta
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import FileResponse, Http404
import threading

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(settings.BASE_DIR)
DATASETS = BASE_DIR / "datasets"
MODELS = BASE_DIR / "models"
MODEL_PATH = MODELS / "face_model.xml"
LABELS_PATH = MODELS / "labels.json"
ATTENDANCE_CSV = MODELS / "attendance.csv"

# Recognizer threshold
RECOGNITION_THRESHOLD = 80.0

# Office Hours
LOG_INTERVAL = timedelta(minutes=2)
MORNING_START = dtime(8, 0)
MORNING_END = dtime(12, 0)
LUNCH_START = dtime(12, 0)
LUNCH_END = dtime(13, 0)
AFTER_START = dtime(13, 0)
AFTER_END = dtime(17, 0)

DATASETS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Cache
_model_lock = threading.Lock()
_recognizer = None
_label_map = None
_last_seen = {}
_last_seen_lock = threading.Lock()


def index(request):
    return render(request, "index.html")


def decode_image(data):
    if not data or "," not in data:
        raise ValueError("Invalid image data")

    try:
        _, enc = data.split(",", 1)
        img_bytes = base64.b64decode(enc, validate=True)
    except (ValueError, binascii.Error) as e:
        raise ValueError("Invalid image data") from e

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img


def _current_period():
    now = datetime.now().time()
    if MORNING_START <= now < MORNING_END:
        return "Morning Presence"
    if LUNCH_START <= now < LUNCH_END:
        return "Lunch Break"
    if AFTER_START <= now < AFTER_END:
        return "Afternoon Presence"
    return None


def _load_model_and_labels():
    global _recognizer, _label_map
    with _model_lock:
        if _recognizer and _label_map:
            return _recognizer, _label_map

        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            raise FileNotFoundError("Model and labels not found; please train first.")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            recognizer.read(str(MODEL_PATH))
        except cv2.error as e:
            raise RuntimeError(f"Failed to load face model: {e}") from e

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            try:
                label_map = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid labels file; please retrain the model.") from e

        _recognizer = recognizer
        _label_map = label_map
        return recognizer, label_map


def _write_attendance(fullname, confidence):
    """fullname is Office:Employee"""
    period = _current_period()
    if not period:
        return

    if ":" not in fullname:
        return

    office, employee = fullname.split(":", 1)

    now = datetime.now()

    with _last_seen_lock:
        last = _last_seen.get(fullname)
        if last and (now - last) < LOG_INTERVAL:
            return
        _last_seen[fullname] = now

    ATTENDANCE_CSV.parent.mkdir(exist_ok=True)
    exists = ATTENDANCE_CSV.exists()

    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "name", "office", "period", "confidence"])
        w.writerow([
            now.strftime("%Y-%m-%d %H:%M:%S"),
            employee,
            office,
            period,
            f"{confidence:.2f}"
        ])


# -------------------------------------------------------------------
# ENROLL EMPLOYEE
# -------------------------------------------------------------------
@csrf_exempt
def enroll(request):
    try:
        data = json.loads(request.body)
        label = data.get("label", "").strip()
        office = data.get("office", "").strip()
        img64 = data.get("image")

        if not label or not office or not img64:
            return JsonResponse({"ok": False, "error": "label, office, and image required"}, status=400)
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid request"}, status=400)

    try:
        img = decode_image(img64)
    except ValueError as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return JsonResponse({"ok": False, "error": "No face detected"})

    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))

    path = DATASETS / office / label
    path.mkdir(parents=True, exist_ok=True)

    count = len(list(path.glob("*.jpg"))) + 1
    save_path = path / f"img_{count:04d}.jpg"
    cv2.imwrite(str(save_path), face)

    return JsonResponse({"ok": True, "saved": str(save_path)})


# -------------------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------------------
@csrf_exempt
def train(request):
    faces = []
    ids = []
    label_map = {}
    cid = 0

    for office in DATASETS.iterdir():
        if not office.is_dir():
            continue
        for employee in office.iterdir():
            if not employee.is_dir():
                continue

            imgs_for_employee = []

            for img_file in employee.glob("*.jpg"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                imgs_for_employee.append(img)

            if not imgs_for_employee:
                continue

            fullname = f"{office.name}:{employee.name}"
            label_map[fullname] = cid
            faces.extend(imgs_for_employee)
            ids.extend([cid] * len(imgs_for_employee))

            cid += 1

    if not faces:
        return JsonResponse({"ok": False, "error": "No training data"})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write(str(MODEL_PATH))

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    global _recognizer, _label_map
    _recognizer = None
    _label_map = None

    return JsonResponse({"ok": True, "labels": label_map})


# -------------------------------------------------------------------
# RECOGNITION
# -------------------------------------------------------------------
@csrf_exempt
def recognize(request):
    try:
        img64 = json.loads(request.body).get("image")
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid request"}, status=400)

    try:
        img = decode_image(img64)
    except ValueError as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return JsonResponse({"ok": True, "results": []})

    try:
        recognizer, label_map = _load_model_and_labels()
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)

    inv = {v: k for k, v in label_map.items()}

    results = []

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label_id, confidence = recognizer.predict(face)

        fullname = inv.get(label_id)
        matched = bool(fullname and ":" in fullname and confidence <= RECOGNITION_THRESHOLD)
        name = fullname.split(":")[1] if matched else "unknown"
        office = fullname.split(":")[0] if matched else ""

        if matched:
            _write_attendance(fullname, confidence)

        results.append({
            "name": name,
            "office": office,
            "confidence": float(confidence),
            "matched": matched,
            "bbox": [int(x), int(y), int(w), int(h)]
        })

    return JsonResponse({"ok": True, "results": results})



def download_attendance(request):
    if not ATTENDANCE_CSV.exists():
        raise Http404("No attendance CSV")
    return FileResponse(open(ATTENDANCE_CSV, "rb"),
                        as_attachment=True,
                        filename="attendance.csv")


# ---------------------- Admin views (safe) -----------------------------------

def admin_offices(request):
    """
    Show list of offices found in attendance CSV (safe if columns missing).
    """
    logs = []
    if ATTENDANCE_CSV.exists():
        with open(ATTENDANCE_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ensure keys exist so later code won't KeyError
                row.setdefault("office", "Unknown")
                logs.append(row)

    # use .get to be safe
    offices = sorted({r.get("office", "Unknown") for r in logs})
    return render(request, "admin_offices.html", {"offices": offices})


def admin_employees(request, office):
    """
    Show unique employee names for a given office.
    """
    logs = []
    if ATTENDANCE_CSV.exists():
        with open(ATTENDANCE_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # set defaults to avoid missing key issues
                row.setdefault("office", "Unknown")
                row.setdefault("name", "")
                if row.get("office", "Unknown") == office:
                    logs.append(row)

    employees = sorted({r.get("name", "") for r in logs if r.get("name")})
    return render(request, "admin_employees.html", {"office": office, "employees": employees})


def admin_employee_logs(request, office, employee):
    """
    Show logs for a specific office+employee, with optional date filter.
    Defensive: uses .get and setdefault to avoid KeyError on missing columns.
    """
    logs = []
    date_filter = request.GET.get("date", "") or datetime.now().strftime("%Y-%m-%d")

    if ATTENDANCE_CSV.exists():
        with open(ATTENDANCE_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ensure presence of keys and normalize
                row.setdefault("office", "Unknown")
                row.setdefault("name", "")
                row.setdefault("timestamp", "")
                row.setdefault("period", "")
                row.setdefault("confidence", "")

                # filter by office and name safely using .get
                if row.get("office", "Unknown") != office or row.get("name", "") != employee:
                    continue

                # filter to the selected date (defaults to today) using YYYY-MM-DD prefix
                ts = row.get("timestamp", "")
                if not ts or not ts.startswith(date_filter):
                    continue

                logs.append(row)

    status = analyze_daily_status(logs)

    return render(request, "admin_employee_logs.html", {
        "office": office,
        "employee": employee,
        "logs": logs,
        "date_filter": date_filter,
        "status": status
    })


def analyze_daily_status(logs):
    """
    Determine if employee is On Time, Late, or Absent for the selected date.
    """

    timestamps = []
    for row in logs:
        ts = row.get("timestamp", "")
        try:
            timestamps.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            continue

    # ABSENT = No valid timestamps at all
    if not timestamps:
        return "Absent"

    # Get earliest log timestamp
    first_time = min(timestamps).time()

    # LATE = First log > 8:00 AM
    if first_time > dtime(8, 0):
        return "Late"

    return "On Time"
