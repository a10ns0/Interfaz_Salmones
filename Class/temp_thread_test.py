import threading
import pyzed.sl as sl # Importa la librería "pesada"
import time

class TestThread(threading.Thread):
    def __init__(self):
        super().__init__()
        print("Constructor OK")

    def run(self):
        print("RUN() OK. Intentando ZED...")
        time.sleep(1) 
        print("RUN() terminada.")

if __name__ == '__main__':
    test_thread = TestThread()
    test_thread.start()
    test_thread.join()
    print("Fin del programa principal.")