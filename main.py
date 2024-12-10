import cv2
import requests
import base64
import json
import tkinter as tk
from PIL import Image, ImageTk
import re
from threading import Thread
from queue import Queue
import logging

# Configuração do logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="log_deteccao_placa.log",
    filemode="w",
)


def recognize_plate_google(image):
    """Envia a imagem para o Google Vision e retorna o texto detectado."""
    logging.info("Iniciando envio ao Google Vision API.")
    try:
        _, buffer = cv2.imencode('.jpg', image)
        content = base64.b64encode(buffer).decode()

        url = 'https://vision.googleapis.com/v1/images:annotate?key=AIzaSyBrittZeYzi2KTE8uXQVqzg7SdQ4DV9oUE'
        headers = {'Content-Type': 'application/json'}
        data = {
            "requests": [
                {
                    "image": {"content": content},
                    "features": [{"type": "TEXT_DETECTION"}]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logging.error(f"Erro na resposta do Google Vision: {response.status_code} - {response.text}")
            return None

        result = response.json()
        logging.debug(f"Resposta do Google Vision: {json.dumps(result, indent=2)}")
        try:
            text = result['responses'][0]['textAnnotations'][0]['description']
            logging.info(f"Texto reconhecido: {text.strip()}")
            return filter_plate_text(text.strip())
        except KeyError:
            logging.warning("Texto não encontrado na resposta do Google Vision.")
            return None
    except Exception as e:
        logging.error(f"Erro ao enviar para o Google Vision: {e}")
        return None


def preprocess_plate(roi):
    """Pré-processa a região da placa sem binarização."""
    logging.info("Pré-processando a região da placa sem binarização.")
    try:
        # Redimensionar para maior resolução
        roi = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2))

        # Aumentar o contraste
        roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=10)

        # Reduzir ruído
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        # Salvar a imagem processada para depuração
        cv2.imwrite("placa_preprocessada.jpg", roi)
        logging.info("Pré-processamento concluído e imagem salva sem binarização.")
        return roi
    except Exception as e:
        logging.error(f"Erro durante o pré-processamento: {e}")
        return roi


def filter_plate_text(text):
    """Filtra o texto para encontrar um padrão de placa brasileira e corrige erros comuns."""
    logging.info(f"Filtrando texto para padrões de placa: {text}")
    matches = re.findall(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', text)
    if matches:
        plate = matches[0]
        # Corrigir erros comuns de OCR
        plate = plate.replace('N', 'W')  # Substituir 'N' por 'W' se necessário
        logging.info(f"Placa válida encontrada: {plate}")
        return plate
    else:
        logging.info("Nenhuma placa válida encontrada no texto.")
        return None


def extract_plate_region(frame, camera_id=1):
    """Define uma região de interesse (ROI) para capturar a placa."""
    logging.info(f"Extraindo ROI da câmera {camera_id}.")
    height, width, _ = frame.shape

    if camera_id == 1:
        # Configuração para câmera 1
        roi_top = int(height * 0.55)
        roi_bottom = int(height * 0.85)
        roi_left = int(width * 0.30)
        roi_right = int(width * 0.70)
    elif camera_id == 2:
        # Configuração para câmera 2
        roi_top = int(height * 0.70)   # Região mais abaixo
        roi_bottom = int(height * 0.95)  # Inclui mais da parte inferior
        roi_left = int(width * 0.30)   # Mais à esquerda
        roi_right = int(width * 0.85)  # Mais à direita
    elif camera_id == 3:
        # Configuração para câmera 3
        roi_top = int(height * 0.60)   # Ajuste específico para a câmera 3
        roi_bottom = int(height * 0.90)  # Ajuste conforme necessário
        roi_left = int(width * 0.25)   # Ajuste conforme necessário
        roi_right = int(width * 0.80)  # Ajuste conforme necessário

    # Extrair a ROI da imagem
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    # Desenhar um retângulo na imagem para verificar visualmente a ROI
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    logging.debug(f"ROI extraída (Câmera {camera_id}): Top={roi_top}, Bottom={roi_bottom}, Left={roi_left}, Right={roi_right}")
    return roi

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Placas - Três Câmeras")

        # Configuração da câmera 1
        self.label_title1 = tk.Label(root, text="Câmera 1", font=("Helvetica", 14, "bold"))
        self.label_title1.grid(row=0, column=0)
        self.video_label1 = tk.Label(root, text="Carregando Câmera 1...", font=("Helvetica", 12))
        self.video_label1.grid(row=1, column=0)
        self.plate_label1 = tk.Label(root, text="Placa Detectada (Câmera 1):", font=("Helvetica", 16))
        self.plate_label1.grid(row=2, column=0)

        # Configuração da câmera 2
        self.label_title2 = tk.Label(root, text="Câmera 2", font=("Helvetica", 14, "bold"))
        self.label_title2.grid(row=0, column=1)
        self.video_label2 = tk.Label(root, text="Carregando Câmera 2...", font=("Helvetica", 12))
        self.video_label2.grid(row=1, column=1)
        self.plate_label2 = tk.Label(root, text="Placa Detectada (Câmera 2):", font=("Helvetica", 16))
        self.plate_label2.grid(row=2, column=1)

        # Configuração da câmera 3
        self.label_title3 = tk.Label(root, text="Câmera 3", font=("Helvetica", 14, "bold"))
        self.label_title3.grid(row=0, column=2)
        self.video_label3 = tk.Label(root, text="Carregando Câmera 3...", font=("Helvetica", 12))
        self.video_label3.grid(row=1, column=2)
        self.plate_label3 = tk.Label(root, text="Placa Detectada (Câmera 3):", font=("Helvetica", 16))
        self.plate_label3.grid(row=2, column=2)

        self.capturing = True
        self.frame_queue1 = Queue(maxsize=1)
        self.frame_queue2 = Queue(maxsize=1)
        self.frame_queue3 = Queue(maxsize=1)

        self.cap1 = None
        self.cap2 = None
        self.cap3 = None

        # Inicia as câmeras
        Thread(target=self.initialize_camera1, daemon=True).start()
        Thread(target=self.initialize_camera2, daemon=True).start()
        Thread(target=self.initialize_camera3, daemon=True).start()

        self.update_video()
    def initialize_camera1(self):
        """Configura a câmera 1."""
        try:
            rtsp_url1 = 'rtsp://admin:lks@123241@192.168.5.80:554/cam/realmonitor?channel=1&subtype=1'
            self.cap1 = cv2.VideoCapture(rtsp_url1)
            if not self.cap1.isOpened():
                logging.error("Erro ao abrir a câmera 1.")
            else:
                logging.info("Câmera 1 carregada com sucesso.")
            Thread(target=self.capture_video, args=(self.cap1, self.frame_queue1, 1), daemon=True).start()
        except Exception as e:
            logging.error(f"Erro na inicialização da câmera 1: {e}")

    def initialize_camera2(self):
        """Configura a câmera 2."""
        try:
            rtsp_url2 = 'rtsp://admin:admin@192.168.5.72:554/'
            self.cap2 = cv2.VideoCapture(rtsp_url2)
            if not self.cap2.isOpened():
                logging.error("Erro ao abrir a câmera 2. Verifique o URL RTSP, usuário e senha.")
            else:
                logging.info("Câmera 2 carregada com sucesso.")
            Thread(target=self.capture_video, args=(self.cap2, self.frame_queue2, 2), daemon=True).start()
        except Exception as e:
            logging.error(f"Erro na inicialização da câmera 2: {e}")

    def initialize_camera3(self):
        """Configura a câmera 3."""
        try:
            rtsp_url3 = 'rtsp://admin:admin@192.168.5.67:554/'  # URL da câmera 3
            self.cap3 = cv2.VideoCapture(rtsp_url3)
            if not self.cap3.isOpened():
                logging.error("Erro ao abrir a câmera 3. Verifique o URL RTSP, usuário e senha.")
            else:
                logging.info("Câmera 3 carregada com sucesso.")
            Thread(target=self.capture_video, args=(self.cap3, self.frame_queue3, 3), daemon=True).start()
        except Exception as e:
            logging.error(f"Erro na inicialização da câmera 3: {e}")

    def capture_video(self, cap, frame_queue, camera_id):
        """Captura frames da câmera e coloca na fila."""
        while self.capturing:
            ret, frame = cap.read()
            if ret:
                if not frame_queue.empty():
                    frame_queue.get()
                frame_queue.put(frame)
            else:
                logging.warning(f"Falha ao capturar frame da câmera {camera_id}.")

    def process_frame(self, frame, plate_label, camera_id):
        """Processa o frame para identificar a placa."""
        logging.info(f"Processando frame da câmera {camera_id}.")
        plate_region = extract_plate_region(frame, camera_id)
        if plate_region is not None:
            processed_plate = preprocess_plate(plate_region)
            plate_text = recognize_plate_google(processed_plate)
            if plate_text:
                plate_label.config(text=f"Placa Detectada: {plate_text}")
                logging.info(f"Câmera {camera_id}: Placa detectada: {plate_text}")
            else:
                plate_label.config(text="Placa Detectada: Nenhuma")
                logging.info(f"Câmera {camera_id}: Nenhuma placa detectada.")
        else:
            plate_label.config(text="Placa Detectada: Nenhuma")
            logging.warning(f"Câmera {camera_id}: Falha ao extrair ROI.")

    def update_video(self):
        """Atualiza os frames das câmeras."""
        # Atualiza vídeo da câmera 1
        if not self.frame_queue1.empty():
            frame1 = self.frame_queue1.get()
            frame1 = cv2.resize(frame1, (640, 480))
            frame_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            img1 = Image.fromarray(frame_rgb1)
            imgtk1 = ImageTk.PhotoImage(image=img1)
            self.video_label1.imgtk = imgtk1
            self.video_label1.configure(image=imgtk1)

            Thread(target=self.process_frame, args=(frame1, self.plate_label1, 1)).start()

        # Atualiza vídeo da câmera 2
        if not self.frame_queue2.empty():
            frame2 = self.frame_queue2.get()
            frame2 = cv2.resize(frame2, (640, 480))
            frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(frame_rgb2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            self.video_label2.imgtk = imgtk2
            self.video_label2.configure(image=imgtk2)

            Thread(target=self.process_frame, args=(frame2, self.plate_label2, 2)).start()

        # Atualiza vídeo da câmera 3
        if not self.frame_queue3.empty():
            frame3 = self.frame_queue3.get()
            frame3 = cv2.resize(frame3, (640, 480))
            frame_rgb3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            img3 = Image.fromarray(frame_rgb3)
            imgtk3 = ImageTk.PhotoImage(image=img3)
            self.video_label3.imgtk = imgtk3
            self.video_label3.configure(image=imgtk3)

            Thread(target=self.process_frame, args=(frame3, self.plate_label3, 3)).start()

        self.root.after(33, self.update_video)

    def __del__(self):
        """Libera recursos ao finalizar."""
        self.capturing = False
        if self.cap1 and self.cap1.isOpened():
            self.cap1.release()
        if self.cap2 and self.cap2.isOpened():
            self.cap2.release()
        if self.cap3 and self.cap3.isOpened():
            self.cap3.release()


# Inicializa a interface
root = tk.Tk()
app = App(root)
root.mainloop()
