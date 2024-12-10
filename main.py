import cv2
import requests
import base64
import json
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import re
from threading import Thread
from queue import Queue
import logging
import time
import os
import json

# Configuração do logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="log_deteccao_placa.log",
    filemode="w",
)

# Variáveis globais para controle
last_sent_time = {1: 0, 2: 0, 3: 0}  # Controle de tempo para envio em cada câmera
FRAME_WIDTH = 640  # Largura padrão para exibição
FRAME_HEIGHT = 360  # Altura padrão para exibição


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
    """Pré-processa a região da placa."""
    try:
        roi = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2))
        roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=10)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        return roi
    except Exception as e:
        logging.error(f"Erro durante o pré-processamento: {e}")
        return roi


def filter_plate_text(text):
    """Filtra o texto para encontrar um padrão de placa brasileira."""
    matches = re.findall(r'[A-Z]{3}[0-9][A-Z][0-9]{2}', text)
    return matches[0] if matches else None


class CameraApp:
    def __init__(self, parent, camera_id, rtsp_url):
        self.parent = parent
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.capturing = True

        self.frame_queue = Queue(maxsize=1)
        self.roi_start = (0, 0)
        self.roi_end = (0, 0)
        self.selecting = False
        self.detected_plate = "Nenhuma"

        # Arquivo para salvar e carregar o ROI
        self.roi_file = f"roi_camera_{self.camera_id}.json"
        self.load_roi()  # Carregar ROI salvo, se existir

        # Frame principal para a câmera
        self.main_frame = tk.Frame(parent, bg="white", borderwidth=2, relief="solid")
        self.main_frame.grid(row=(camera_id - 1) // 2, column=(camera_id - 1) % 2, padx=5, pady=5, sticky="nsew")

        # Rótulo de vídeo
        self.video_label = tk.Label(self.main_frame, bg="black", text=f"Carregando câmera {camera_id}...", font=("Arial", 10), fg="white")
        self.video_label.pack(padx=5, pady=5, fill="both", expand=True)

        # Rótulo da placa detectada
        self.plate_label = tk.Label(
            self.main_frame, text=f"Câmera {camera_id}: Nenhuma placa detectada.", font=("Arial", 12), fg="green", bg="white"
        )
        self.plate_label.pack(pady=5)

        # Vincular eventos do mouse para seleção de ROI
        self.video_label.bind("<ButtonPress-1>", self.start_select)
        self.video_label.bind("<B1-Motion>", self.update_select)
        self.video_label.bind("<ButtonRelease-1>", self.end_select)

        # Iniciar a captura da câmera em uma nova thread
        Thread(target=self.initialize_camera, daemon=True).start()
        self.update_video()


    def initialize_camera(self):
        """Configura a câmera."""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                logging.error(f"Erro ao abrir a câmera {self.camera_id}.")
            else:
                logging.info(f"Câmera {self.camera_id} carregada com sucesso.")
            Thread(target=self.capture_video, daemon=True).start()
        except Exception as e:
            logging.error(f"Erro na inicialização da câmera {self.camera_id}: {e}")

    def capture_video(self):
        """Captura frames da câmera."""
        global last_sent_time

        while self.capturing:
            ret, frame = self.cap.read()
            if ret:
                # Redimensionar frame para tamanho padrão
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # Desenhar ROI
                if self.roi_start != (0, 0) and self.roi_end != (0, 0):
                    cv2.rectangle(frame, self.roi_start, self.roi_end, (0, 255, 0), 2)

                    # Processar a ROI a cada 1 segundo
                    current_time = time.time()
                    if current_time - last_sent_time[self.camera_id] > 1:  # 1 segundo
                        x1, y1 = self.roi_start
                        x2, y2 = self.roi_end
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            plate = recognize_plate_google(preprocess_plate(roi))
                            if plate:
                                self.detected_plate = plate
                                self.plate_label.config(text=f"Placa Detectada (Câmera {self.camera_id}): {self.detected_plate}")
                            else:
                                self.plate_label.config(text=f"Placa Detectada (Câmera {self.camera_id}): Nenhuma")
                        last_sent_time[self.camera_id] = current_time

                # Atualizar feed
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)



    def save_roi(self):
        """Salva as coordenadas do ROI em um arquivo JSON."""
        roi_data = {"start": self.roi_start, "end": self.roi_end}
        with open(self.roi_file, "w") as file:
            json.dump(roi_data, file)
        logging.info(f"ROI salvo para a câmera {self.camera_id}: {roi_data}")

    def load_roi(self):
        """Carrega as coordenadas do ROI de um arquivo JSON, se existir."""
        if os.path.exists(self.roi_file):
            with open(self.roi_file, "r") as file:
                roi_data = json.load(file)
                self.roi_start = tuple(roi_data.get("start", (0, 0)))
                self.roi_end = tuple(roi_data.get("end", (0, 0)))
                logging.info(f"ROI carregado para a câmera {self.camera_id}: {roi_data}")

    def start_select(self, event):
        """Inicia a seleção da ROI."""
        self.roi_start = (event.x, event.y)
        self.selecting = True

    def update_select(self, event):
        """Atualiza a seleção da ROI enquanto arrasta."""
        self.roi_end = (event.x, event.y)

    def end_select(self, event):
        """Finaliza a seleção da ROI."""
        self.roi_end = (event.x, event.y)
        self.selecting = False
        self.save_roi()  # Salvar ROI ao final da seleção
        logging.info(f"ROI selecionada (Câmera {self.camera_id}): Início={self.roi_start}, Fim={self.roi_end}")

    def update_video(self):
        """Atualiza os frames da câmera."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()

            # Desenhar ROI carregada, se existir
            if self.roi_start != (0, 0) and self.roi_end != (0, 0):
                cv2.rectangle(frame, self.roi_start, self.roi_end, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.parent.after(33, self.update_video)


# Inicializa a interface principal
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detecção de Placas - Multicâmeras")
        self.geometry("1200x800")
        self.configure(bg="white")

        # Caminho do arquivo de configuração
        self.config_file = "cameras.json"
        self.cameras_config = self.load_camera_config()

        # Menu lateral estilizado
        self.sidebar = tk.Frame(self, bg="#283747", width=250)
        self.sidebar.pack(side="left", fill="y")

        # Adicionar logo ao menu
        self.logo_image = Image.open("retaguarda.png")
        self.logo_image = self.logo_image.resize((150, 150), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        self.logo_label = tk.Label(self.sidebar, image=self.logo_photo, bg="#283747")
        self.logo_label.pack(pady=20)

        # Botão para expandir/recolher o menu
        self.toggle_button = tk.Button(
            self.sidebar,
            text="≡",
            bg="#1C2833",
            fg="white",
            font=("Arial", 16, "bold"),
            command=self.toggle_menu,
            relief="flat",
        )
        self.toggle_button.pack(pady=15, padx=10)

        # Botão Configurar câmera
        self.config_button = tk.Button(
            self.sidebar,
            text="Configurar Câmera",
            bg="#117A65",
            fg="white",
            font=("Arial", 14, "bold"),
            command=self.configure_camera,
            relief="flat",
        )
        self.config_button.pack(pady=10, padx=20, fill="x")

        # Botão Gerenciar Câmeras
        self.manage_button = tk.Button(
            self.sidebar,
            text="Gerenciar Câmeras",
            bg="#D35400",
            fg="white",
            font=("Arial", 14, "bold"),
            command=self.manage_cameras,
            relief="flat",
        )
        self.manage_button.pack(pady=10, padx=20, fill="x")

        # Frame principal
        self.main_frame = tk.Frame(self, bg="white")
        self.main_frame.pack(side="right", fill="both", expand=True)

        # Tornar a grade principal responsiva
        for i in range(2):  # 2 linhas
            self.main_frame.grid_rowconfigure(i, weight=1)
        for j in range(2):  # 2 colunas
            self.main_frame.grid_columnconfigure(j, weight=1)

        # Carregar as câmeras configuradas
        self.cameras = []
        self.load_cameras()

    def toggle_menu(self):
        """Expande ou recolhe o menu lateral."""
        if self.sidebar.winfo_ismapped():
            self.sidebar.pack_forget()  # Recolher o menu
            self.icon_button.lift()  # Mostrar o botão de ícone
        else:
            self.sidebar.pack(side="left", fill="y")  # Expandir o menu
            self.icon_button.lower()  # Esconder o botão de ícone

    def configure_camera(self):
        """Abre a janela para configurar uma nova câmera."""
        config_window = tk.Toplevel(self)
        config_window.title("Configurar Câmera")
        config_window.geometry("400x200")
        config_window.configure(bg="white")

        # Campo para selecionar o número da câmera
        tk.Label(config_window, text="Número da Câmera:", bg="white", font=("Arial", 12)).pack(pady=5)
        camera_number = ttk.Combobox(config_window, values=["1", "2", "3", "4", "5", "6"], font=("Arial", 12))
        camera_number.pack(pady=5)

        # Campo para inserir o caminho da câmera
        tk.Label(config_window, text="URL da Câmera (RTSP):", bg="white", font=("Arial", 12)).pack(pady=5)
        camera_url = tk.Entry(config_window, width=40, font=("Arial", 12))
        camera_url.pack(pady=5)

        def save_configuration():
            cam_number = camera_number.get()
            cam_url = camera_url.get()

            if not cam_number or not cam_url:
                messagebox.showerror("Erro", "Por favor, preencha todos os campos!")
                return

            try:
                # Criar uma nova instância de CameraApp para a nova câmera
                self.cameras.append(CameraApp(self.main_frame, int(cam_number), cam_url))
                self.cameras_config.append({"camera_id": int(cam_number), "url": cam_url})
                self.save_camera_config()

                messagebox.showinfo("Sucesso", f"Câmera {cam_number} configurada com sucesso!")
                config_window.destroy()

            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao configurar a câmera: {e}")

        save_button = tk.Button(
            config_window, text="Salvar Configuração", bg="#117A65", fg="white", font=("Arial", 12), command=save_configuration
        )
        save_button.pack(pady=20)

    def manage_cameras(self):
        """Abre uma janela para gerenciar câmeras (remover)."""
        manage_window = tk.Toplevel(self)
        manage_window.title("Gerenciar Câmeras")
        manage_window.geometry("400x300")
        manage_window.configure(bg="white")

        tk.Label(manage_window, text="Câmeras Configuradas:", bg="white", font=("Arial", 14)).pack(pady=10)

        camera_listbox = tk.Listbox(manage_window, font=("Arial", 12))
        camera_listbox.pack(pady=10, padx=10, fill="both", expand=True)

        # Preencher a lista com câmeras
        for cam in self.cameras_config:
            camera_listbox.insert(tk.END, f"Câmera {cam['camera_id']} - {cam['url']}")

        def remove_selected():
            selected = camera_listbox.curselection()
            if not selected:
                messagebox.showerror("Erro", "Selecione uma câmera para remover!")
                return

            index = selected[0]
            del self.cameras_config[index]
            self.cameras[index].main_frame.destroy()
            del self.cameras[index]

            self.save_camera_config()
            camera_listbox.delete(index)

            messagebox.showinfo("Sucesso", "Câmera removida com sucesso!")

        remove_button = tk.Button(
            manage_window, text="Remover Câmera", bg="#C0392B", fg="white", font=("Arial", 12), command=remove_selected
        )
        remove_button.pack(pady=10)

    def load_camera_config(self):
        """Carrega as configurações de câmeras de um arquivo JSON."""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as file:
                return json.load(file)
        return []

    def save_camera_config(self):
        """Salva as configurações de câmeras em um arquivo JSON."""
        with open(self.config_file, "w") as file:
            json.dump(self.cameras_config, file)

    def load_cameras(self):
        """Carrega as câmeras configuradas na interface."""
        for cam in self.cameras_config:
            self.cameras.append(CameraApp(self.main_frame, cam["camera_id"], cam["url"]))


# Inicializa o aplicativo
if __name__ == "__main__":
    app = App()
    app.mainloop()



