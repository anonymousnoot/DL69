# การบ้านสัปดาห์ที่ 11 — Timeseries LSTM Dashboard

สร้าง Django dashboard สำหรับทำนายข้อมูลอนุกรมเวลาด้วย LSTM (SSE streaming)

## วิธีรัน

```bash
uv sync
uv run manage.py runserver
```

เปิด http://127.0.0.1:8000/

## สิ่งที่นักศึกษาต้องทำ

1. **`dashboard/ml/train.py`** — implement `event_stream()` ให้สมบูรณ์
   (โหลดข้อมูล, sliding window, LSTM, train/val split, บันทึก prediction plot)
2. **`dashboard/views.py`** — เปลี่ยน `student_id` และ `student_name` เป็นของจริง

## ข้อมูล

- `data/temperature.csv` — ข้อมูลอุณหภูมิจำลอง (2,000 ชั่วโมง) พร้อมใช้แล้ว
- สามารถใช้ข้อมูลอนุกรมเวลาชุดอื่นได้ตามต้องการ (วางใน `data/`)

## หมายเหตุ

- Python 3.13 (กำหนดใน `pyproject.toml`)
- ส่งงานผ่าน branch `wk11`
