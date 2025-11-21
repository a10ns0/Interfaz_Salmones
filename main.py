

import cv2


from Class.SalmonApp import SalmonApp


if __name__ == "__main__":
    # Crear un archivo de video falso si no existe (para pruebas)
    from pathlib import Path
    if not Path("video_salmones.svo").exists():
        print("Creando archivo de video de marcador de posición 'video_salmon.mp4'...")
        import numpy as np
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('video_salmon.mp4', fourcc, 20.0, (640, 480))
        for i in range(100): # 100 fotogramas
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = (i * 2, 50, 0) # Fondo cambiante
            cv2.putText(frame, f'Frame {i}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
            2)
            out.write(frame)
        out.release()
        print("Video de marcador de posición creado.")
app = SalmonApp()
app.mainloop()
