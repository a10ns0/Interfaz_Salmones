
import sys
import cv2
import math
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# --- CONFIGURACIÓN DE ARCHIVO ---
SVO_PATH = "video_salmones.svo"  # <--- COLOCA AQUÍ LA RUTA DE TU ARCHIVO

# Para el LARGO (Length)
# Usamos la cabeza (0) y la base de la cola (2) que es más estable que las puntas de las aletas.
KEYPOINT_HEAD = 0   # "Head"
KEYPOINT_TAIL = 2   # "Tail"

# Para el ANCHO/ALTO (Width/Height)
# La parte más ancha suele ser entre la aleta dorsal (4) y la pélvica (5).
KEYPOINT_TOP = 4    # "Dorsal_fin"
KEYPOINT_BOTTOM = 5 # "Pelvic_fin"


"""

Función get_3d_distance:

Esta función encapsula la lógica de consultar a la ZED. Le das dos píxeles 2D (ej. nariz y cola) y ella consulta la point_cloud para obtener la posición real en el espacio.

Manejo de errores: Devuelve valid=False si uno de los puntos está fuera de rango, muy cerca de la cámara (zona muerta) o es un reflejo inválido (NaN).

"""
def get_3d_distance(p1_pixel, p2_pixel, point_cloud):
    """ Calcula distancia euclidiana 3D entre dos pixeles """
    u1, v1 = int(p1_pixel[0]), int(p1_pixel[1])
    u2, v2 = int(p2_pixel[0]), int(p2_pixel[1])

    # Verificar que los puntos estén dentro de la imagen
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

def detections_to_custom_box(results, im_width, im_height):
    output = []
    if results[0].boxes:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
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



"""
Correlación (Match) ID vs Keypoints:

Este es el paso más importante. El SDK de ZED te devuelve los objetos con ID, pero pierde la información de "cuál keypoint es cuál" que tenía YOLO.
La solución implementada (líneas 144-160) compara el centro de la caja del objeto rastreado por ZED con las cajas que detectó YOLO en ese instante. 
Si están muy cerca (distancia < 50 px), asumimos que son el mismo pez y recuperamos sus puntos clave para medir.

"""


def main():
    # 1. CARGAR MODELO YOLO
    print("Cargando modelo YOLO Pose...")
    model = YOLO('best.pt') 

    # 2. INICIAR ZED CON VIDEO SVO
    print(f"Abriendo archivo SVO: {SVO_PATH}")
    zed = sl.Camera()
    
    # Configuración de Input
    input_type = sl.InputType()
    input_type.set_from_svo_file(SVO_PATH) # <--- AQUÍ SE CARGA EL VIDEO

    init_params = sl.InitParameters(input_t=input_type)
    init_params.coordinate_units = sl.UNIT.METER 
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    
    # IMPORTANTE: Desactiva el modo tiempo real para procesar cada frame sin saltos
    # (útil si el procesamiento de IA es pesado)
    init_params.svo_real_time_mode = False 

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error al abrir el archivo SVO: {err}")
        exit()

    # 3. CONFIGURAR TRACKING
    detection_parameters = sl.ObjectDetectionParameters()
    detection_parameters.enable_tracking = True
    detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    
    zed.enable_positional_tracking()
    zed.enable_object_detection(detection_parameters)

    objects = sl.Objects()
    image_zed = sl.Mat()
    point_cloud = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    key = ''
    print("Procesando video... Presiona 'q' para salir.")

    while key != 113: # 113 es 'q'
        err = zed.grab(runtime_params)
        
        if err == sl.ERROR_CODE.SUCCESS:
            # A. Obtener datos del frame actual del video
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
            
            frame_np = image_zed.get_data()
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)
            height, width = frame_rgb.shape[:2]

            # B. Inferencia YOLO
            results = model(frame_rgb, verbose=False)
            
            # C. Ingestión al Tracking
            custom_boxes = detections_to_custom_box(results, width, height)
            zed.ingest_custom_box_objects(custom_boxes)

            # D. Recuperar objetos trackeados
            if zed.retrieve_objects(objects, sl.ObjectDetectionRuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
                
                # Correlacionar ID ZED con Keypoints YOLO por distancia
                if results[0].boxes and results[0].keypoints:
                    yolo_boxes = results[0].boxes.xywh.cpu().numpy()
                    yolo_kpts = results[0].keypoints.data.cpu().numpy()

                    for obj in objects.object_list:
                        zed_center_x = (obj.bounding_box_2d[0][0] + obj.bounding_box_2d[2][0]) / 2
                        zed_center_y = (obj.bounding_box_2d[0][1] + obj.bounding_box_2d[2][1]) / 2

                        best_match_idx = -1
                        min_dist = 99999

                        for i, box in enumerate(yolo_boxes):
                            dist = math.hypot(box[0] - zed_center_x, box[1] - zed_center_y)
                            if dist < 50: 
                                min_dist = dist
                                best_match_idx = i
                        
                        if best_match_idx != -1:
                            kpts = yolo_kpts[best_match_idx]
                            
                            # Medir Largo
                            p_head, p_tail = kpts[KEYPOINT_HEAD], kpts[KEYPOINT_TAIL]
                            valid_l, largo_m = False, 0.0
                            if p_head[2] > 0.5 and p_tail[2] > 0.5:
                                valid_l, largo_m = get_3d_distance(p_head[:2], p_tail[:2], point_cloud)

                            # Medir Ancho
                            p_top, p_bottom = kpts[KEYPOINT_TOP], kpts[KEYPOINT_BOTTOM]
                            valid_w, ancho_m = False, 0.0
                            if p_top[2] > 0.5 and p_bottom[2] > 0.5:
                                valid_w, ancho_m = get_3d_distance(p_top[:2], p_bottom[:2], point_cloud)

                            # Visualización
                            top_left = (int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1]))
                            bottom_right = (int(obj.bounding_box_2d[2][0]), int(obj.bounding_box_2d[2][1]))
                            
                            cv2.rectangle(frame_np, top_left, bottom_right, (0, 255, 255), 2)
                            
                            dims_text = f"ID:{obj.id} "
                            if valid_l: dims_text += f"L:{largo_m*100:.1f}cm "
                            if valid_w: dims_text += f"W:{ancho_m*100:.1f}cm"

                            cv2.putText(frame_np, dims_text, (top_left[0], top_left[1]-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # Líneas visuales
                            if valid_l: cv2.line(frame_np, (int(p_head[0]), int(p_head[1])), (int(p_tail[0]), int(p_tail[1])), (0,0,255), 2)
                            if valid_w: cv2.line(frame_np, (int(p_top[0]), int(p_top[1])), (int(p_bottom[0]), int(p_bottom[1])), (255,0,0), 2)

            cv2.imshow("Analisis SVO Salmones", frame_np)
            key = cv2.waitKey(10) # Pequeña espera para permitir refresco de ventana
            
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("Fin del video SVO alcanzado.")
            # Opcion A: Salir
            break
            # Opcion B: Repetir video
            # zed.set_svo_position(0) 

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()