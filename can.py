import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import cv2
import easyocr
import time
import re
import threading
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import deque

# ----------------- CONFIGURAÇÕES -----------------
MODEL_PATH = "license_plate_detector.pt"  
CAMERA_URL = "rtsp://admin:OFni@120@169.254.41.91:554/Streaming/Channels/102"
INTERVALO_OCR = 0.2     
NUM_THREADS = 2        
LOG_LIMIT = 10          

# ----------------- INICIALIZAÇÃO -----------------
reader = easyocr.Reader(['pt', 'en'], gpu=False)
model = YOLO(MODEL_PATH)

ultima_placa = ""
placa_busca = ""
ocr_queue = queue.Queue()
lock_ultima_placa = threading.Lock()
log_leituras = deque(maxlen=LOG_LIMIT)

# ----------------- FUNÇÕES -----------------
def placa_valida_brasil(texto):
    return bool(re.match(r'^[A-Z]{3}[0-9]{4}$', texto) or re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', texto))

def atualizar_log(log_box):
    log_box.delete(0, tk.END)
    for leitura in list(log_leituras)[::-1]:
        log_box.insert(tk.END, leitura)

def processar_ocr_worker(status_label, log_box):
    global ultima_placa, placa_busca
    while True:
        placa_crop = ocr_queue.get()
        if placa_crop is None:
            break

        resultados = reader.readtext(placa_crop)
        if resultados:
            texto_placa = max(resultados, key=lambda r: r[2])[1]
            texto_placa = re.sub(r'[^A-Z0-9]', '', texto_placa.upper())

            if placa_valida_brasil(texto_placa):
                with lock_ultima_placa:
                    ultima_placa = texto_placa
                    cor = "green" if texto_placa == placa_busca else "orange"
                    msg = f"{'✅' if cor=='green' else '⚠️'} {texto_placa}"
                    status_label.after(0, lambda: status_label.config(text=msg, foreground=cor))

                    log_leituras.append(f"[{time.strftime('%H:%M:%S')}] {texto_placa}")
                    log_box.after(0, lambda: atualizar_log(log_box))

        ocr_queue.task_done()

# ----------------- CAPTURA SEM DELAY -----------------
def captura_camera_thread(camera_url, frame_queue):
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # força descartar frames antigos

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mantém apenas o último frame (evita acumular e gerar delay)
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                pass

        frame_queue.put(frame)

# ----------------- LOOP PRINCIPAL -----------------
def iniciar_camera(label_video, status_label, label_placa_img, log_box):
    frame_queue = queue.Queue(maxsize=1)
    threading.Thread(target=captura_camera_thread, args=(CAMERA_URL, frame_queue), daemon=True).start()

    last_ocr_time = 0

    def atualizar_imagem(label, frame_bgr, size):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(frame_rgb).resize(size)
        im_tk = ImageTk.PhotoImage(im_pil)
        label.imgtk = im_tk
        label.config(image=im_tk)

    def loop_video():
        nonlocal last_ocr_time
        if frame_queue.empty():
            root.after(10, loop_video)
            return

        frame = frame_queue.get()

        # Reduz para YOLO rápido
        small_frame = cv2.resize(frame, (640, 360))
        results = model(small_frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            escala_x = frame.shape[1] / small_frame.shape[1]
            escala_y = frame.shape[0] / small_frame.shape[0]
            x1, y1, x2, y2 = int(x1*escala_x), int(y1*escala_y), int(x2*escala_x), int(y2*escala_y)

            placa_crop = frame[y1:y2, x1:x2]
            atualizar_imagem(label_placa_img, placa_crop, (200, 60))

            if time.time() - last_ocr_time > INTERVALO_OCR:
                ocr_queue.put(placa_crop)
                last_ocr_time = time.time()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        atualizar_imagem(label_video, frame, (800, 450))
        root.after(30, loop_video)  # ~33 FPS

    loop_video()

# ----------------- INTERFACE -----------------
root = tk.Tk()
root.title("🚗 Leitor de Placas - Sem Delay")
root.configure(bg="#f5f5f5")

frame_top = tk.Frame(root, bg="#f5f5f5")
frame_top.pack(pady=10)

ttk.Label(frame_top, text="Digite a placa para buscar:").grid(row=0, column=0, padx=5)
entrada_placa = ttk.Entry(frame_top, font=("Arial", 16), width=12)
entrada_placa.grid(row=0, column=1, padx=5)

def atualizar_placa_busca():
    global placa_busca
    placa_busca = re.sub(r'[^A-Z0-9]', '', entrada_placa.get().strip().upper())
    status_label.config(text=f"Buscando placa: {placa_busca}", foreground="blue")

ttk.Button(frame_top, text="Atualizar", command=atualizar_placa_busca).grid(row=0, column=2, padx=5)

status_label = tk.Label(root, text="Nenhuma placa buscada ainda.", font=("Arial", 14), bg="#f5f5f5")
status_label.pack(pady=10)

frame_main = tk.Frame(root, bg="#f5f5f5")
frame_main.pack()

label_video = tk.Label(frame_main, bg="#000000")
label_video.grid(row=0, column=0, padx=10)

log_box = tk.Listbox(frame_main, width=40, height=20, font=("Courier", 12))
log_box.grid(row=0, column=1, padx=10, sticky="ns")

frame_bottom = tk.Frame(root, bg="#f5f5f5")
frame_bottom.pack(pady=10)
tk.Label(frame_bottom, text="Última placa detectada:", font=("Arial", 12), bg="#f5f5f5").pack()
label_placa_img = tk.Label(frame_bottom, bg="#ffffff", relief="solid", bd=1)
label_placa_img.pack(pady=5)

for _ in range(NUM_THREADS):
    threading.Thread(target=processar_ocr_worker, args=(status_label, log_box), daemon=True).start()

iniciar_camera(label_video, status_label, label_placa_img, log_box)
root.mainloop()
