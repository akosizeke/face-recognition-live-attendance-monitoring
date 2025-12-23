from django.shortcuts import render
import os
import shutil
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
# Start morning presence at 5:00 AM so early arrivals are counted as on time
MORNING_START = dtime(5, 0)
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
_match_counts = {}


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


def _safe_employee_dir(office, label):
    office = (office or "").strip()
    label = (label or "").strip()
    if not office or not label:
        return None

    base = DATASETS.resolve()
    path = (DATASETS / office / label).resolve()
    if base not in path.parents:
        return None
    return path


def _extract_primary_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    face_gray = gray[y:y+h, x:x+w]
    if face_gray.size == 0:
        return None, None
    face_gray = cv2.resize(face_gray, (200, 200))
    face_gray = cv2.equalizeHist(face_gray)

    face_color = img[y:y+h, x:x+w]
    if face_color.size == 0:
        face_color = None
    else:
        face_color = cv2.resize(face_color, (240, 240))

    return face_gray, face_color


def _save_profile_image(path, face_gray, face_color):
    profile_path = path / "profile.jpg"
    if face_color is not None:
        cv2.imwrite(str(profile_path), face_color)
    else:
        cv2.imwrite(str(profile_path), face_gray)
    return profile_path


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

    face_gray, face_color = _extract_primary_face(img)
    if face_gray is None:
        return JsonResponse({"ok": False, "error": "No face detected"})

    path = _safe_employee_dir(office, label)
    if path is None:
        return JsonResponse({"ok": False, "error": "Invalid office or label"}, status=400)
    path.mkdir(parents=True, exist_ok=True)

    count = len(list(path.glob("img_*.jpg"))) + 1
    save_path = path / f"img_{count:04d}.jpg"
    cv2.imwrite(str(save_path), face_gray)

    profile_path = path / "profile.jpg"
    if not profile_path.exists():
        _save_profile_image(path, face_gray, face_color)

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

            img_files = sorted(employee.glob("img_*.jpg"))
            if not img_files:
                img_files = [p for p in employee.glob("*.jpg") if p.name != "profile.jpg"]

            for img_file in img_files:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                img = cv2.equalizeHist(img)
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
        payload = json.loads(request.body)
        img64 = payload.get("image")
        threshold_val = float(payload.get("threshold", RECOGNITION_THRESHOLD) or RECOGNITION_THRESHOLD)
        threshold_val = max(30.0, min(150.0, threshold_val))  # clamp to sane range
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

    now = datetime.now()

    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        face = cv2.equalizeHist(face)
        label_id, confidence = recognizer.predict(face)

        fullname = inv.get(label_id)
        matched = bool(fullname and ":" in fullname and confidence <= threshold_val)

        # Stabilize across frames: require at least 2 consecutive hits per label within 3s
        state = _match_counts.get(label_id, {"count": 0, "ts": now})
        if (now - state["ts"]) > timedelta(seconds=3):
            state["count"] = 0
        state["count"] += 1 if matched else 0
        state["ts"] = now
        _match_counts[label_id] = state

        stable = matched and state["count"] >= 2

        name = fullname.split(":")[1] if stable else "unknown"
        office = fullname.split(":")[0] if stable else ""

        if stable:
            _write_attendance(fullname, confidence)

        results.append({
            "name": name,
            "office": office,
            "confidence": float(confidence),
            "matched": stable,
            "bbox": [int(x), int(y), int(w), int(h)]
        })

    return JsonResponse({"ok": True, "results": results})



def profile_image(request, office, employee):
    path = _safe_employee_dir(office, employee)
    if path is None:
        raise Http404("Invalid profile path")

    profile_path = path / "profile.jpg"
    if profile_path.exists():
        return FileResponse(open(profile_path, "rb"), content_type="image/jpeg")

    img_files = sorted(path.glob("img_*.jpg"))
    if not img_files:
        img_files = [p for p in path.glob("*.jpg") if p.name != "profile.jpg"]
        img_files = sorted(img_files)

    if img_files:
        return FileResponse(open(img_files[0], "rb"), content_type="image/jpeg")

    raise Http404("Profile not found")


@csrf_exempt
def update_profile(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        label = (data.get("label") or "").strip()
        office = (data.get("office") or "").strip()
        img64 = data.get("image")
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid request"}, status=400)

    if not label or not office or not img64:
        return JsonResponse({"ok": False, "error": "label, office, and image required"}, status=400)

    try:
        img = decode_image(img64)
    except ValueError as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    face_gray, face_color = _extract_primary_face(img)
    if face_gray is None:
        return JsonResponse({"ok": False, "error": "No face detected"})

    path = _safe_employee_dir(office, label)
    if path is None:
        return JsonResponse({"ok": False, "error": "Invalid office or label"}, status=400)
    path.mkdir(parents=True, exist_ok=True)

    count = len(list(path.glob("img_*.jpg"))) + 1
    save_path = path / f"img_{count:04d}.jpg"
    cv2.imwrite(str(save_path), face_gray)

    profile_path = _save_profile_image(path, face_gray, face_color)

    return JsonResponse({"ok": True, "saved": str(save_path), "profile": str(profile_path)})


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
    offices = set()

    if DATASETS.exists():
        for office_dir in DATASETS.iterdir():
            if office_dir.is_dir():
                offices.add(office_dir.name)

    logs = []
    if ATTENDANCE_CSV.exists():
        with open(ATTENDANCE_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ensure keys exist so later code won't KeyError
                row.setdefault("office", "Unknown")
                logs.append(row)

    # use .get to be safe
    for row in logs:
        offices.add(row.get("office", "Unknown"))
    offices = sorted(offices)
    return render(request, "admin_offices.html", {"offices": offices})


def admin_employees(request, office):
    """
    Show unique employee names for a given office.
    """
    employees = set()

    office_dir = DATASETS / office
    if office_dir.exists() and office_dir.is_dir():
        for employee_dir in office_dir.iterdir():
            if employee_dir.is_dir():
                employees.add(employee_dir.name)

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

    for row in logs:
        name = row.get("name", "")
        if name:
            employees.add(name)
    employees = sorted(employees)
    return render(request, "admin_employees.html", {"office": office, "employees": employees})


def admin_employee_logs(request, office, employee):
    """
    Show logs for a specific office+employee, with optional date filter.
    Defensive: uses .get and setdefault to avoid KeyError on missing columns.
    """
    all_logs = []
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

                all_logs.append(row)

    # filter to the selected date (defaults to today) using YYYY-MM-DD prefix
    logs = []
    for row in all_logs:
        ts = row.get("timestamp", "")
        if ts and ts.startswith(date_filter):
            logs.append(row)

    status = analyze_daily_status(logs)
    late_days = _collect_late_days(all_logs)

    return render(request, "admin_employee_logs.html", {
        "office": office,
        "employee": employee,
        "logs": logs,
        "date_filter": date_filter,
        "status": status,
        "late_days": late_days
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


def _collect_late_days(rows):
    """
    Build a summary of dates where the first log was after 8:00 AM.
    """
    first_by_date = {}

    for row in rows:
        ts = row.get("timestamp", "")
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        date_key = dt.strftime("%Y-%m-%d")
        current = first_by_date.get(date_key)
        if not current or dt < current["dt"]:
            first_by_date[date_key] = {
                "dt": dt,
                "period": row.get("period", ""),
                "confidence": row.get("confidence", "")
            }

    late_days = []
    for date_key, info in first_by_date.items():
        if info["dt"].time() > dtime(8, 0):
            late_days.append({
                "date": date_key,
                "time": info["dt"].strftime("%H:%M:%S"),
                "period": info.get("period", ""),
                "confidence": info.get("confidence", "")
            })

    late_days.sort(key=lambda r: r["date"], reverse=True)
    return late_days


# ---------------------- API helpers -----------------------------------------

def list_offices(request):
    """
    Return a consolidated list of offices based on dataset folders and attendance CSV.
    """
    offices = set()

    # Collect from datasets directory (trained/enrolled offices)
    if DATASETS.exists():
        for office_dir in DATASETS.iterdir():
            if office_dir.is_dir():
                offices.add(office_dir.name)

    # Collect from attendance logs (offices that already have records)
    if ATTENDANCE_CSV.exists():
        with open(ATTENDANCE_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("office") or ""
                if name:
                    offices.add(name)

    return JsonResponse({"ok": True, "offices": sorted(offices)})


@csrf_exempt
def delete_office(request):
    """
    Delete an office: removes dataset folder and prunes attendance rows.
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Method not allowed"}, status=405)

    try:
        office = (json.loads(request.body).get("office") or "").strip()
    except Exception:
        office = ""

    if not office:
        return JsonResponse({"ok": False, "error": "Office required"}, status=400)

    # Remove dataset directory if present
    office_dir = DATASETS / office
    if office_dir.exists() and office_dir.is_dir():
        shutil.rmtree(office_dir, ignore_errors=True)

    # Rewrite attendance CSV without that office
    if ATTENDANCE_CSV.exists():
        tmp_path = ATTENDANCE_CSV.with_suffix(".tmp")
        with open(ATTENDANCE_CSV, newline='', encoding='utf-8') as src, \
                open(tmp_path, "w", newline='', encoding='utf-8') as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("office") != office:
                    writer.writerow(row)
        tmp_path.replace(ATTENDANCE_CSV)

    # reset in-memory caches to avoid stale labels
    global _recognizer, _label_map
    _recognizer = None
    _label_map = None

    return JsonResponse({"ok": True, "deleted": office})


@csrf_exempt
def rename_office(request):
    """
    Rename an office: renames dataset folder, updates attendance rows, and relabels keys.
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except Exception:
        data = {}

    old = (data.get("old") or "").strip()
    new = (data.get("new") or "").strip()

    if not old or not new:
        return JsonResponse({"ok": False, "error": "Both old and new office names are required"}, status=400)
    if old == new:
        return JsonResponse({"ok": False, "error": "Office name unchanged"}, status=400)

    dst_dir = DATASETS / new
    if dst_dir.exists():
        return JsonResponse({"ok": False, "error": "New office name already exists"}, status=400)

    # rename dataset folder if it exists
    src_dir = DATASETS / old
    if src_dir.exists() and src_dir.is_dir():
        shutil.move(str(src_dir), str(dst_dir))

    # rewrite attendance CSV with updated office
    if ATTENDANCE_CSV.exists():
        tmp_path = ATTENDANCE_CSV.with_suffix(".tmp")
        with open(ATTENDANCE_CSV, newline='', encoding='utf-8') as src, \
                open(tmp_path, "w", newline='', encoding='utf-8') as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get("office") == old:
                    row["office"] = new
                writer.writerow(row)
        tmp_path.replace(ATTENDANCE_CSV)

    # update labels.json if present (rename office prefix)
    if LABELS_PATH.exists():
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = {}

        updated = {}
        for k, v in labels.items():
            if k.startswith(f"{old}:"):
                updated[f"{new}:{k.split(':', 1)[1]}"] = v
            else:
                updated[k] = v

        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)

    global _recognizer, _label_map
    _recognizer = None
    _label_map = None

    return JsonResponse({"ok": True, "old": old, "new": new})
