#!/usr/bin/env python3
import sys, os, struct
import numpy as np

# Загрузка onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    # Попробуем путь из virtualenv VideoT16
    venv_site = '/home/work/VideoT16/.venv/lib/python3.12/site-packages'
    if os.path.exists(venv_site):
        sys.path.insert(0, venv_site)
        import onnxruntime as ort
    else:
        sys.stderr.write("onnxruntime not found\n")
        sys.exit(1)

model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/anti_uav_fast_320_int8_pc.onnx'
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
sys.stderr.write(f"[ONNX_WORKER] Модель загружена: {model_path}\n")
sys.stderr.flush()

# Размер входного тензора [1, 3, 320, 320] float32 = 1 * 3 * 320 * 320 * 4 = 1,228,800 байт
TENSOR_BYTES = 1 * 3 * 320 * 320 * 4

# Сигнал готовности
sys.stdout.write("READY\n")
sys.stdout.flush()

while True:
    try:
        raw = sys.stdin.buffer.read(TENSOR_BYTES)
        if len(raw) < TENSOR_BYTES:
            break

        tensor = np.frombuffer(raw, dtype=np.float32).reshape((1, 3, 320, 320))
        out = sess.run(None, {'images': tensor})[0] # [1, 6, 2100]

        # Парсинг кандидатов с порогом 8% (как в проверенном проекте)
        scores = np.maximum(out[0, 4, :], out[0, 5, :])
        mask = scores >= 0.08
        
        detections = []
        if np.any(mask):
            indices = np.where(mask)[0]
            for idx in indices:
                cx, cy, w, h = out[0, 0:4, idx]
                conf = scores[idx]
                class_id = 0 if out[0, 4, idx] >= out[0, 5, idx] else 1
                detections.append((float(cx), float(cy), float(w), float(h), float(conf), int(class_id)))

        # Отправляем количество и детекции в бинарном виде: [count: u32, (cx: f32, cy: f32, w: f32, h: f32, conf: f32, class_id: u32)...]
        count = len(detections)
        header = struct.pack('<I', count)
        sys.stdout.buffer.write(header)
        for d in detections:
            sys.stdout.buffer.write(struct.pack('<fffffI', d[0], d[1], d[2], d[3], d[4], d[5]))
        sys.stdout.buffer.flush()

    except Exception as e:
        sys.stderr.write(f"[ONNX_WORKER ERROR] {e}\n")
        break
