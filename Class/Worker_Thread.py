import threading
import queue
import time

# --- NUEVAS IMPORTACIONES ---
import pyzed.sl as sl
import cv2
import numpy as np


"""
Este es el diseño arquitectónico de la clase VideoProcessingThread. Esta clase hereda de
threading.Thread y encapsula toda la lógica de procesamiento.

"""

class VideoProcessingThread(threading.Thread):
    
    
    def __init__(self, video_path, data_queue, pause_event, stop_event):
    
        """

        Constructor del hilo trabajador.
        :param video_path: Ruta al archivo de video.
        :param data_queue: queue.Queue para enviar datos a la GUI.
        :param pause_event: threading.Event para controlar la pausa/reanudación.
        :param stop_event: threading.Event para señalar la detención.
        
        """

        super().__init__()
        print(" [DEBUG] Hilo trabajador: Constructor ejecutado.")
        self.video_path = video_path
        self.data_queue = data_queue
        self.pause_event = pause_event
        self.stop_event = stop_event
        
        # El modelo se cargará DENTRO de este hilo, no en la GUI.
        # (Ver Sección 6 para más detalles)
        #self.model = None


    def load_model(self):
        """
        Método de marcador de posición para cargar el modelo de estimación.
        Esto puede tardar varios segundos.
        """
        print(" Cargando modelo...")
        
        # En un caso real:
        # self.model = torch.load('modelo_salmon.pth')
        # self.model.eval()
        
        print(" Modelo cargado.")
        return "Modelo_Cargado" # Marcador de posición
    


    def process_frame(self, frame):
        
        """
        Método de marcador de posición para el procesamiento del modelo.
        (Ver Sección 6 para la implementación real)
        """

        # El frame que recibes aquí es BGR (después de la corrección en poll_queue).
        # Trabaja en HSV para aislar el brillo/contraste sin tocar el color.
        
        # --- 1. AISLAR CANAL DE LUMINOSIDAD (V de HSV) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        h, s, v = cv2.split(hsv) # Separar canales H, S, V (Luminosidad)

        # 2. Aplicar Ecualización de Histograma Adaptativa (CLAHE) solo al canal V (brillo)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced_v = clahe.apply(v)
        
        # 3. Combinar los canales originales de color (H, S) con el canal V mejorado
        v_combined_hsv = cv2.merge([h, s, contrast_enhanced_v])
        
        # 4. Convertir de nuevo a BGR (Este es el frame de color, pero con contraste mejorado)
        processed_frame = cv2.cvtColor(v_combined_hsv, cv2.COLOR_HSV2BGR) 
        
        # --- 5. AUMENTO DE NITIDEZ (Sharpening) EN COLOR ---
        # Aplicamos el filtro de nitidez (Sharpening) al frame de color mejorado.
        kernel = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]]) 
        sharpened_frame = cv2.filter2D(processed_frame, -1, kernel)
        # Aquí es donde se llama al modelo real:
        # estimaciones = self.model.predict(frame)
        # Marcador de posición para datos de estimación
        estimaciones = {"dimensiones": (10, 5)}
        # Aquí se dibujarían las estimaciones en el fotograma:
        # processed_frame = self.draw_estimations_on_frame(frame, estimaciones)
        return sharpened_frame, estimaciones # Devolvemos el fotograma original por ahora    
    

    def run(self):
            """
            El método principal del hilo. Este es el bucle de procesamiento.

            """
                
            print(" [DEBUG] Hilo: RUN() INICIADA.")

            # --- BLOQUE DE PRUEBA 1: CARGA DEL MODELO ---
            print(" [DEBUG] Hilo: A punto de cargar modelo.")
            self.model = self.load_model()
            print(" [DEBUG] Hilo: Modelo cargado con éxito.")
            # --- [INICIO DE LA MODIFICACIÓN PARA .SVO] ---
            # 1. Inicializar la cámara ZED
            zed = sl.Camera() 
            init_parameters = sl.InitParameters()
            

        # --- MODIFICACIÓN CLAVE ---
            # Cambiar a modo NONE para deshabilitar completamente el cómputo de profundidad (y la IA)
            init_parameters.depth_mode = sl.DEPTH_MODE.NONE
            # 2. Configurar para usar el archivo SVO en lugar de la cámara en vivo
            init_parameters.set_from_svo_file(self.video_path) 
            
            # Configurar para reproducir a la máxima velocidad posible (no a tiempo real)
            init_parameters.svo_real_time_mode = False 
            
            # Contenedor de imagen ZED (matriz de datos)
            image_zed = sl.Mat()
            
            # Intentar abrir el archivo SVO
            status = zed.open(init_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f" Error: No se pudo abrir el archivo SVO {self.video_path}. Código: {status}")
                zed.close()
                self.data_queue.put(None) # Señal de fin
                return
            # --- [FIN DE LA MODIFICACIÓN PARA .SVO: INICIALIZACIÓN] ---
            
            try:
                # Reemplazamos 'cap' con 'zed'
                while not self.stop_event.is_set():
                    # ... PUNTO DE CONTROL DE PAUSA sin cambios ...
                    self.pause_event.wait()
                    
                    # --- [INICIO DE LA MODIFICACIÓN PARA .SVO: LECTURA] ---
                    # Intentar grabar el siguiente fotograma del SVO
                    if zed.grab() == sl.ERROR_CODE.SUCCESS:
                        # Obtener la imagen estéreo izquierda (VIEW.LEFT)
                        # Esto copia los datos del frame en 'image_zed'
                        zed.retrieve_image(image_zed, sl.VIEW.LEFT) 
                        
                        # Convertir sl.Mat a un array NumPy. 
                        # El slice [:, :, 0:3] se asegura de obtener 3 canales (RGB)
                        # incluso si la ZED SDK usa 4 (RGBA/BGRA) por defecto.
                        frame = image_zed.get_data()[:, :, 0:3]

                    else:
                        # Si grab() no fue exitoso, puede ser el final del archivo.
                        if zed.get_svo_position() >= zed.get_svo_number_of_frames() - 1:
                            print(" Fin del archivo SVO.")
                            break # Sale del bucle
                        continue # Continúa con la siguiente iteración si hubo un error temporal
                    # --- [FIN DE LA MODIFICACIÓN PARA .SVO: LECTURA] ---

                    # ... Procesamiento del frame y envío a la cola sin cambios ...
                    processed_frame, estimations = self.process_frame(frame)
                    
                    output_data = {
                        "frame": processed_frame,
                        "dimensions": estimations["dimensiones"]
                    }
                    
                    try:
                        self.data_queue.put(output_data, timeout=1)
                    except queue.Full:
                        pass
            finally:
                # Asegurarse de que la cámara ZED se cierre.
                if 'zed' in locals() and zed.is_opened():
                    zed.close()
                # Señalar a la GUI que hemos terminado
                self.data_queue.put(None)
                print(" Hilo de procesamiento detenido.")



"""

Cuando se quiera volver a trabajar con el modelo de estimación, 
se recomienda volver a activar el modo sl.DEPTH_MODE.NEURAL o sl.DEPTH_MODE.ULTRA 
y dejar que la optimización de 30 o mas minutos se complete.


"""