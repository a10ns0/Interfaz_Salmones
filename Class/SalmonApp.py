"""
La siguiente clase SalmonApp se construye sobre customtkinter.CTk y servirá como la ventana
principal de la aplicación.

"""

import customtkinter
import cv2

from PIL import Image, ImageTk
import queue
import threading
# Importar la clase VideoProcessingThread
from Class.Worker_Thread import VideoProcessingThread

# Configuración de CustomTkinter
customtkinter.set_appearance_mode("Dark") # Modos: "System", "Dark", "Light"
customtkinter.set_default_color_theme("green") # Temas: "blue", "green", "dark-blue"


class SalmonApp(customtkinter.CTk):
    def __init__(self):
       
        super().__init__()
        
        
        # --- 1. Configuración de la Ventana Principal ---
        self.title("Estimador de Dimensiones de Salmones")
        self.geometry("1024x768")
        
        
        
        # --- 2. Variables de Estado y Control de Hilos ---
        self.data_queue = queue.Queue(maxsize=10) # Cola para recibir datos del trabajador
        self.pause_event = threading.Event() # Evento para pausar/reanudar
        self.stop_event = threading.Event() # Evento para detener el hilo
        self.video_thread = None # Contenedor para el objeto del hilo
        # Ruta al video (marcador de posición)
        self.video_path = "video_salmones.svo" # Reemplazar con la ruta real
        
        
        
        # --- 3. Creación de Widgets de la GUI ---
        # Marco principal
        self.main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)
        # Widget de Video (un CTkLabel que mostrará las imágenes)
        # Este es el "lienzo" para el video.
        self.video_label = customtkinter.CTkLabel(self.main_frame, text="El video se mostrará aquí",
        width=800, height=600) 
        self.video_label.pack(pady=20, padx=20, fill="both", expand=True)
        # demuestra el uso de un Label para mostrar un feed de webcam.
        # Marco de Controles
        self.controls_frame = customtkinter.CTkFrame(self.main_frame)
        self.controls_frame.pack(fill="x", padx=20, pady=(0, 20))

        
        # Botón Start
        self.start_button = customtkinter.CTkButton(self.controls_frame, text="Start",
        command=self.start_video_processing)
        self.start_button.pack(side="left", fill="x", expand=True, padx=5)
        
        # Botón Pausa
        self.pause_button = customtkinter.CTkButton(self.controls_frame, text="Pausa",
        command=self.pause_video_processing)
        self.pause_button.pack(side="left", fill="x", expand=True, padx=5)
        self.pause_button.configure(state="disabled") # Deshabilitado hasta que comience
        
        # Etiqueta de estado (para mostrar las dimensiones)
        self.status_label = customtkinter.CTkLabel(self.controls_frame, text="Dimensiones: N/A",
        anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True, padx=10)
        
        
        
        # --- 4. Manejo del Cierre de la Ventana ---
        # Asignar una función personalizada al botón 'X' de la ventana
        self.protocol("WM_DELETE_WINDOW", self.on_closing)





    def start_video_processing(self):
        """Callback para el botón Start."""
        if self.video_thread is not None and self.video_thread.is_alive():
            # Si el hilo ya está vivo, solo reanudamos
            print(" Reanudando procesamiento...")
            self.pause_event.set()
        else:
            # Si el hilo no existe o ha muerto, crear uno nuevo
            print(" Iniciando nuevo procesamiento...")
            self.stop_event.clear() # Asegurarse de que el evento de parada esté limpio
            self.pause_event.set() # Establecer en 'set' para que se ejecute inmediatamente
            self.video_thread = VideoProcessingThread(
                video_path=self.video_path,
                data_queue=self.data_queue,
                pause_event=self.pause_event,
                stop_event=self.stop_event
            )
            print(" Iniciando el hilo trabajador...")
            self.video_thread.start() # Iniciar el hilo
            # Iniciar el bucle de sondeo de la cola
            self.poll_queue()

        # Actualizar estado de los botones
        self.start_button.configure(text="Reanudar")
        self.pause_button.configure(state="normal")


    def pause_video_processing(self):
        """Callback para el botón Pausa."""
        print(" Pausando procesamiento...")
        self.pause_event.clear() # Señaliza al hilo trabajador que debe pausar



    def poll_queue(self):
        """
        Sondea la cola de datos en busca de nuevos fotogramas del hilo trabajador.
        Este es el corazón de la actualización de la GUI.
        """


        try:
            # Intentar obtener datos de la cola SIN BLOQUEAR
            output_data = self.data_queue.get_nowait()
        except queue.Empty:
            # La cola está vacía, no hay nada que hacer.
            # Volver a programar esta función para que se ejecute de nuevo.
            self.after(20, self.poll_queue) # 20ms ≈ 50 FPS de sondeo
        else:
            # ¡Recibimos datos!
            if output_data is None:
                # Señal de "Fin de video" del trabajador
                print(" Se recibió la señal de fin de video.")
                self.start_button.configure(text="Start") # Restablecer botón
                self.pause_button.configure(state="disabled")
                return # No reprogramar el sondeo
            
            # --- 1. Extraer Datos ---
            frame = output_data["frame"]
            dimensions = output_data["dimensions"] # (Tu error de Pylance desaparece cuando usas 'dimensions' abajo)
            
            # --- 2. Conversión Crítica: Numpy a CustomTkinter (ImageTk) ---
            
            # 2a. ANTERIORMENTE (ASUMIENDO OpenCV BGR): 
            cv2_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2b. Array de Numpy (RGB) a Imagen PIL.
            pil_img = Image.fromarray(cv2_img_rgb)
            # 2c. Redimensionar la imagen PIL (Opcional, pero recomendado)
            # para que se ajuste al widget CTkLabel sin cambiar la geometría.
            # (Se asume que video_label tiene 800x600, se puede usar.winfo_width())
            pil_img_resized = pil_img.resize((800, 600), Image.LANCZOS)
            # 2d. Imagen PIL a Imagen Tkinter.
            self.tk_image = ImageTk.PhotoImage(image=pil_img_resized)
            # --- 3. Actualizar Widgets de la GUI ---
            # 3a. Actualizar el widget de video (el CTkLabel)
            self.video_label.configure(image=self.tk_image, text="")
            
            # 3b. Actualizar la etiqueta de dimensiones
            self.status_label.configure(text=f"Dimensiones: {dimensions[0]} x {dimensions[1]}")

            # ¡¡CRÍTICO!! Guardar una referencia a la imagen.
            # Tkinter/CustomTkinter es propenso a que el recolector de basura
            # elimine la imagen si no se mantiene una referencia.
            self.video_label.image = self.tk_image
            # --- 4. Volver a programar el Sondeo ---
            # Programar esta misma función para que se ejecute de nuevo
            # después de 'X' milisegundos.
            self.after(20, self.poll_queue)


    def on_closing(self):
        """Se llama cuando el usuario cierra la ventana."""
        print(" Cierre solicitado. Deteniendo el hilo trabajador...")
        # 1. Señalizar al hilo trabajador que debe detenerse
        self.stop_event.set()
        self.pause_event.set() # Desbloquearlo de 'wait()' si está pausado
        # 2. Esperar a que el hilo trabajador termine limpiamente
        if self.video_thread is not None:
            try:
                self.video_thread.join(timeout=2.0) # Esperar máx 2 seg
                print(" Hilo trabajador detenido limpiamente.")
            except Exception as e:
                print(f" Excepción al hacer join: {e}")
        # 3. Destruir la ventana de la GUI
        self.destroy()