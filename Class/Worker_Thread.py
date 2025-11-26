import threading
import queue
import time
import math
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO


import pandas as pd # <--- NUEVA IMPORTACIÓN
from datetime import datetime # Para nombrar el archivo con fecha
# --- CONFIGURACIÓN DE KEYPOINTS ---
# (Mismos índices que en tu test.py)
KEYPOINT_HEAD = 0
KEYPOINT_TAIL = 2
KEYPOINT_TOP = 4
KEYPOINT_BOTTOM = 5

class VideoProcessingThread(threading.Thread):
    
    def __init__(self, video_path, data_queue, pause_event, stop_event):
        super().__init__()
        print(" [DEBUG] Hilo trabajador: Constructor ejecutado.")
        self.video_path = video_path
        self.data_queue = data_queue
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.model = None
        self.zed = None
        self.objects = None
        self.detection_runtime_params = None

        # NUEVO: Lista para acumular datos
        self.fish_data_log = [] 
        self.frame_count = 0 # Contador de frames

    
    
    
    def load_model(self):
        print(" Cargando modelo YOLO...")
        # NOTA: Usamos 'best.pt' para coincidir con tu test.py
        try:
            model = YOLO('best.pt') 
        except:
            print(" [ADVERTENCIA] No se encontró 'best.pt', usando 'yolov8n-pose.pt' por defecto.")
            model = YOLO('yolov8n-pose.pt')
            
        print(" Modelo cargado.")
        return model

    def init_zed_tracking(self, zed):
        """ Configura el tracking usando CUSTOM_BOX_OBJECTS """
        print(" Inicializando Tracking ZED...")
        detection_parameters = sl.ObjectDetectionParameters()
        detection_parameters.enable_tracking = True
        # detection_parameters.enable_mask_output = False # BORRADO (Causaba error)
        detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS

        zed.enable_positional_tracking()
        err = zed.enable_object_detection(detection_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f" ERROR al habilitar detección/tracking: {err}")
        
        self.objects = sl.Objects()
        self.detection_runtime_params = sl.ObjectDetectionRuntimeParameters()

    def detections_to_custom_box(self, results, im_width, im_height):
        """ Convierte detecciones YOLO a formato ZED """
        output = []
        if results[0].boxes:
            boxes = results[0].boxes
            for box in boxes:
                xywh = box.xywh[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                x_c, y_c, w, h = xywh
                x_min = x_c - (w / 2)
                y_min = y_c - (h / 2)
                x_max = x_c + (w / 2)
                y_max = y_c + (h / 2)

                zed_obj = sl.CustomBoxObjectData()
                zed_obj.bounding_box_2d = np.array([
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max]
                ])
                zed_obj.label = cls
                zed_obj.probability = conf
                zed_obj.is_grounded = False 
                output.append(zed_obj)
        return output

    def get_3d_distance(self, p1_pixel, p2_pixel, point_cloud):
        """ Lógica idéntica a tu test.py """
        u1, v1 = int(p1_pixel[0]), int(p1_pixel[1])
        u2, v2 = int(p2_pixel[0]), int(p2_pixel[1])

        height, width = point_cloud.get_height(), point_cloud.get_width()
        if not (0 <= u1 < width and 0 <= v1 < height and 0 <= u2 < width and 0 <= v2 < height):
            return False, 0.0

        err1, point3D_1 = point_cloud.get_value(u1, v1)
        err2, point3D_2 = point_cloud.get_value(u2, v2)

        if err1 == sl.ERROR_CODE.SUCCESS and err2 == sl.ERROR_CODE.SUCCESS:
            x1, y1, z1 = point3D_1[0], point3D_1[1], point3D_1[2]
            x2, y2, z2 = point3D_2[0], point3D_2[1], point3D_2[2]

            if math.isfinite(x1) and math.isfinite(x2):
                distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                return True, distancia
        
        return False, 0.0

   
   
   
   
    def process_frame(self, frame, point_cloud):
        # 1. MEJORA DE IMAGEN (Tu código original de HSV/CLAHE/Sharpen)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced_v = clahe.apply(v)
        v_combined_hsv = cv2.merge([h, s, contrast_enhanced_v])
        processed_frame = cv2.cvtColor(v_combined_hsv, cv2.COLOR_HSV2BGR) 
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) 
        
        # Esta es la imagen sobre la que dibujaremos
        sharpened_frame = cv2.filter2D(processed_frame, -1, kernel)

        # 2. INFERENCIA YOLO
        results = self.model(sharpened_frame, verbose=False)
        
        dimensions_output = (0.0, 0.0)

        # 3. TRACKING Y VISUALIZACIÓN (Lógica portada de test.py)
        if results[0].boxes and results[0].keypoints:
            
            # A. Ingestión
            custom_boxes = self.detections_to_custom_box(results, frame.shape[1], frame.shape[0])
            self.zed.ingest_custom_box_objects(custom_boxes)

            # B. Recuperación de objetos trackeados
            if self.zed.retrieve_objects(self.objects, self.detection_runtime_params) == sl.ERROR_CODE.SUCCESS:
                
                # Datos crudos de YOLO para correlacionar
                yolo_boxes = results[0].boxes.xywh.cpu().numpy()
                yolo_kpts = results[0].keypoints.data.cpu().numpy()

                # Iterar sobre cada objeto que ZED está siguiendo
                for obj in self.objects.object_list:
                    
                    # --- LÓGICA DE CORRELACIÓN DE test.py ---
                    zed_center_x = (obj.bounding_box_2d[0][0] + obj.bounding_box_2d[2][0]) / 2
                    zed_center_y = (obj.bounding_box_2d[0][1] + obj.bounding_box_2d[2][1]) / 2

                    best_match_idx = -1
                    min_dist = 99999

                    # Buscamos qué detección de YOLO corresponde a este ID de ZED
                    for i, box in enumerate(yolo_boxes):
                        dist = math.hypot(box[0] - zed_center_x, box[1] - zed_center_y)
                        if dist < 50: # Umbral de coincidencia
                            min_dist = dist
                            best_match_idx = i
                    
                    # Si encontramos el match, tenemos los keypoints y el ID juntos
                    if best_match_idx != -1:
                        kpts = yolo_kpts[best_match_idx]

                        # --- CÁLCULOS ---
                        # Medir Largo
                        p_head = kpts[KEYPOINT_HEAD]
                        p_tail = kpts[KEYPOINT_TAIL]
                        valid_l, largo_m = False, 0.0
                        if p_head[2] > 0.5 and p_tail[2] > 0.5:
                            valid_l, largo_m = self.get_3d_distance(p_head[:2], p_tail[:2], point_cloud)

                        # Medir Ancho
                        p_top = kpts[KEYPOINT_TOP]
                        p_bottom = kpts[KEYPOINT_BOTTOM]
                        valid_w, ancho_m = False, 0.0
                        if p_top[2] > 0.5 and p_bottom[2] > 0.5:
                            valid_w, ancho_m = self.get_3d_distance(p_top[:2], p_bottom[:2], point_cloud)
                        
                        # Guardar para enviar a GUI (solo del último detectado por ahora)
                        if valid_l:
                            largo_cm = largo_m * 100
                            ancho_cm = ancho_m * 100 if valid_w else 0.0

                            # Calcular peso estimado
                            peso_g = self.calculate_weight(largo_cm)

                            dimensions_output = (largo_cm, ancho_cm)
                            # --- NUEVO: GUARDAR EN EL REGISTRO (PANDAS) ---
                            data_entry = {
                                "Frame": self.frame_count,
                                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                                "ID_Salmon": obj.id,
                                "Largo_cm": round(largo_cm, 2),
                                "Ancho_cm": round(ancho_cm, 2),
                                "Peso_Estimado_g": round(peso_g, 2),
                                "Confianza_Deteccion": round(obj.confidence, 2),
                            }
                            self.fish_data_log.append(data_entry)



                        # --- DIBUJAR EN FRAME (Visualización) ---
                        
                        top_left = (int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1]))
                        bottom_right = (int(obj.bounding_box_2d[2][0]), int(obj.bounding_box_2d[2][1]))
                        
                        # 1. Caja Amarilla
                        cv2.rectangle(sharpened_frame, top_left, bottom_right, (0, 255, 255), 2)

                        # 2. Texto ID y Dimensiones
                        dims_text = f"ID:{obj.id} "
                        if valid_l: dims_text += f"L:{largo_m*100:.1f}cm "
                        if valid_w: dims_text += f"W:{ancho_m*100:.1f}cm"

                        cv2.putText(sharpened_frame, dims_text, (top_left[0], top_left[1]-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # 3. Líneas sobre el cuerpo (Rojo Largo, Azul Ancho)
                        if valid_l: 
                            cv2.line(sharpened_frame, (int(p_head[0]), int(p_head[1])), 
                                     (int(p_tail[0]), int(p_tail[1])), (0,0,255), 2)
                        if valid_w: 
                            cv2.line(sharpened_frame, (int(p_top[0]), int(p_top[1])), 
                                     (int(p_bottom[0]), int(p_bottom[1])), (255,0,0), 2)


        return sharpened_frame, {"dimensiones": dimensions_output}

    
    
    
    
    
    
    def calculate_weight(self, length_cm):
            """
            Calcula el peso basado en el Factor de Condición de Fulton (K).
            Formula: W = (K / 100) * L^3
            
            Referencias para K en Salmo salar:
            - 1.00 - 1.10: Salmón salvaje o post-desove (Flaco).
            - 1.20 - 1.30: Salmón de cultivo promedio.
            - 1.30 - 1.45: Salmón de cultivo calidad Premium (Gordo/Robusto).
            """
            
            # Factor de Condición (Ajustable según la realidad del centro de cultivo)
            K_factor = 1.36 
            
            # Convertimos K a 'a' y asumimos isometría (b=3)
            a = K_factor / 100
            b = 3.0
            
            weight_g = a * (length_cm ** b)
            return weight_g
    







# --- 2. MÉTODO PARA GUARDAR CON FILTRO DE CALIDAD ---
    def save_data_to_file(self):
        """ 
        Genera el reporte, pero FILTRA los peces que aparecieron muy poco tiempo
        (ruido o falsos positivos).
        """
        if not self.fish_data_log:
            print(" [INFO] No hay datos para generar reporte.")
            return

        print(" [INFO] Procesando datos y eliminando ruido...")
        df = pd.DataFrame(self.fish_data_log)
        
        # 1. Agrupar por ID
        grouped = df.groupby('ID_Salmon')
        
        # 2. FILTRO: El salmón debe haber sido detectado en al menos N frames
        # (Por ejemplo, 5 frames, aprox 0.2 segundos si vas a 30fps)
        MIN_FRAMES_THRESHOLD = 5 
        
        valid_ids = grouped.filter(lambda x: len(x) >= MIN_FRAMES_THRESHOLD)

        if valid_ids.empty:
            print(" [INFO] Se detectaron objetos, pero ninguno superó el umbral de permanencia (ruido).")
            return

        # 3. Generar Resumen de los peces válidos
        summary_df = valid_ids.groupby('ID_Salmon').agg({
            'Largo_cm': 'mean',
            'Ancho_cm': 'mean',
            'Peso_Estimado_g': 'mean',
            'Confianza_Deteccion': 'mean',
            'Frame': 'count',                 # Cuántos frames duró
        })
        
        # Renombrar columnas para que se vea bonito en Excel
        summary_df.columns = ['Largo_Prom_cm', 'Ancho_Prom_cm', 'Peso_Prom_g', 
                              'Confianza_Prom', 'Total_Frames']

        # 4. Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_raw = f"registro_completo_{timestamp}.csv"
        filename_summary = f"resumen_final_peces_{timestamp}.csv"
        
        valid_ids.to_csv(filename_raw, index=False)
        summary_df.to_csv(filename_summary)
        
        print(f" [EXITO] Reporte generado: {len(summary_df)} salmones válidos identificados.")
        print(f"         Archivos: {filename_summary} (Resumen) y {filename_raw} (Detalle)")





    def run(self):
        print(" [DEBUG] Hilo: RUN() INICIADA.")

        self.model = self.load_model()
        
        self.zed = sl.Camera()
        init_parameters = sl.InitParameters()
        init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL 
        init_parameters.coordinate_units = sl.UNIT.METER
        init_parameters.set_from_svo_file(self.video_path)
        init_parameters.svo_real_time_mode = False 
        
        print(f" Abriendo SVO: {self.video_path}")
        status = self.zed.open(init_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f" Error: No se pudo abrir SVO. Código: {status}")
            self.data_queue.put(None)
            return

        self.init_zed_tracking(self.zed)

        image_zed = sl.Mat()
        point_cloud = sl.Mat()

        self.frame_count = 0 # Reiniciar contador

        try:
            while not self.stop_event.is_set():
                self.pause_event.wait()
                
                if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                    
                    self.frame_count += 1 # <--- INCREMENTAR CONTADOR

                    self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

                    frame_bgra = image_zed.get_data()
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

                    processed_frame, estimations = self.process_frame(frame_bgr, point_cloud)
                    
                    output_data = {
                        "frame": processed_frame,
                        "dimensions": estimations["dimensiones"]
                    }
                    
                    try:
                        self.data_queue.put(output_data, timeout=1)
                    except queue.Full:
                        pass
                else:
                    if self.zed.get_svo_position() >= self.zed.get_svo_number_of_frames() - 1:
                        print(" Fin del archivo SVO.")
                        break

        finally:
            if self.zed.is_opened():
                self.zed.close()
            
            # --- NUEVO: GUARDAR DATOS AL TERMINAR ---
            self.save_data_to_file()

            self.data_queue.put(None)
            print(" Hilo detenido.")



"""

Cuando se quiera volver a trabajar con el modelo de estimación, 
se recomienda volver a activar el modo sl.DEPTH_MODE.NEURAL o sl.DEPTH_MODE.ULTRA 
y dejar que la optimización de 30 o mas minutos se complete.


"""



"""

Al cerrar el programa o terminar el video, se generarán dos archivos CSV en la carpeta de tu proyecto:

1)
registros_completos_fecha.csv: Un registro fotograma a fotograma. Ideal para ver la evolución y limpiar ruido después.

Ej: ID 1, Frame 10, Largo 50.1

Ej: ID 1, Frame 11, Largo 50.2


2)
resumen_por_pez_fecha.csv: Una tabla limpia con un salmón por fila.

Ej: ID 1 | Largo Promedio: 50.15 cm | Peso Promedio: 1.5 kg | Visto en: 45 frames

"""