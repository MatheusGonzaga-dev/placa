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



def extract_plate_region(frame):
    """Define uma região de interesse (ROI) maior para capturar a placa."""
    logging.info("Extraindo região de interesse (ROI) da placa.")
    try:
        height, width, _ = frame.shape

        # Ajustar os limites da ROI para aumentar o campo de detecção
        roi_top = int(height * 0.55)  # Inclui mais área acima
        roi_bottom = int(height * 0.85)  # Inclui mais área abaixo
        roi_left = int(width * 0.30)  # Inclui mais área à esquerda
        roi_right = int(width * 0.70)  # Inclui mais área à direita

        # Extrair a ROI
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]

        # Desenhar o retângulo na imagem para visualização
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

        logging.info("ROI extraída com sucesso.")
        return roi
    except Exception as e:
        logging.error(f"Erro ao extrair ROI: {e}")
        return None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Placas - Câmera 1")

        self.video_label = tk.Label(root, text="Carregando Câmera 1...", font=("Helvetica", 12))
        self.video_label.grid(row=0, column=0)

        self.plate_label = tk.Label(root, text="Placa Detectada:", font=("Helvetica", 16))
        self.plate_label.grid(row=1, column=0)

        self.capturing = True
        self.frame_queue = Queue(maxsize=1)

        self.cap = None
        self.last_processed_plate = None  # Armazena a última imagem da placa processada
        Thread(target=self.initialize_camera, daemon=True).start()

        self.update_video()

    def initialize_camera(self):
        """Configura a câmera."""
        try:
            rtsp_url = 'rtsp://admin:lks@123241@192.168.5.80:554/cam/realmonitor?channel=1&subtype=1'
            self.cap = cv2.VideoCapture(rtsp_url)
            if not self.cap.isOpened():
                logging.error("Erro ao abrir a câmera.")
            else:
                logging.info("Câmera carregada com sucesso.")
            Thread(target=self.capture_video, args=(self.cap, self.frame_queue), daemon=True).start()
        except Exception as e:
            logging.error(f"Erro na inicialização da câmera: {e}")

    def capture_video(self, cap, frame_queue):
        """Captura frames da câmera e coloca na fila."""
        while self.capturing:
            ret, frame = cap.read()
            if ret:
                if not frame_queue.empty():
                    frame_queue.get()
                frame_queue.put(frame)

    def process_frame(self, frame):
        """Processa o frame para identificar a placa."""
        logging.info("Processando frame para detecção de placa.")
        plate_region = extract_plate_region(frame)
        if plate_region is not None:
            # Evita processar a mesma imagem várias vezes
            if self.last_processed_plate is not None:
                difference = cv2.absdiff(self.last_processed_plate, plate_region)
                mean_diff = cv2.mean(difference)[0]  # Calcula a diferença média entre os frames

                # Define um limiar para processar novamente somente se a placa mudou
                if mean_diff < 20:  # Ajuste o valor conforme necessário
                    logging.info("Imagem da placa é similar à anterior, ignorando processamento.")
                    return

            # Atualiza a última placa processada
            self.last_processed_plate = plate_region.copy()

            processed_plate = preprocess_plate(plate_region)

            plate_text = recognize_plate_google(processed_plate)
            if plate_text:
                self.plate_label.config(text=f"Placa Detectada: {plate_text}")
            else:
                self.plate_label.config(text="Placa Detectada: Nenhuma")
        else:
            self.plate_label.config(text="Placa Detectada: Nenhuma")

    def update_video(self):
        """Atualiza os frames da câmera."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            Thread(target=self.process_frame, args=(frame,)).start()

        self.root.after(33, self.update_video)

    def __del__(self):
        """Libera recursos ao finalizar."""
        self.capturing = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


root = tk.Tk()
app = App(root)
root.mainloop()
