Readme

🫁UAONeumonia
Herramienta para la detección rápida de neumonía en radiografías de tórax en formato DICOM utilizando Deep Learning.

---

Descripción

Este proyecto implementa una Red Neuronal Convolucional (CNN) para clasificar radiografías de tórax en tres categorías:

1. Neumonía Bacteriana  
2. Neumonía Viral  
3. Sin Neumonía  

Además, integra la técnica de explicabilidad **Grad-CAM**, que genera un mapa de calor sobre la imagen para resaltar las regiones relevantes utilizadas por el modelo para tomar la decisión.

---

Flujo del sistema

1. Carga de imagen DICOM  
2. Lectura y extracción del arreglo de imagen  
3. Preprocesamiento ( normalización)  
4. Inferencia del modelo CNN  
5. Generación de Grad-CAM  
6. Visualización en la interfaz  
7. Exportación opcional (CSV / PDF)

---

Instalación

### 1️⃣ Clonar el repositorio

git clone https://github.com/heyjuanes/uaoneumonia.git
cd uaoneumonia

### 2️⃣ Clonar el repositorio

python -m venv venv

### 3️⃣ Activar entorno (Windows PowerShell)

.\venv\Scripts\Activate.ps1

### 4️⃣ Instalar dependencias

pip install -r requirements.txt

### ▶️ Ejecución

La aplicación se ejecuta desde:
python ui/main.py

Es una aplicación web (Flask), abrir en navegador:

http://127.0.0.1:5000/

-----------

Uso de aplicativo 

1. Ingresar el nombre y el numero de cedula del paciente
2. Seleccione o arrastre la imagen del explorador de archivos 
3. Oprima el boton analizar y espere unos segundos hasta que observe los resultados (Diagnostico,Mapa de calor y la confianza del modelo).
4. Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv
5. Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz
6. Oprima nueva imegen si quiere analizar un nuevo paciente.



<img width="896" height="594" alt="image" src="https://github.com/user-attachments/assets/f0514756-eb0e-43b7-a1d8-b75d440f1c77" />

Fuente_Elabolacion propia

--------

Descripcion de Modulos
1. *Read_img.py:*Lee la imagen en formato DICOM utilizando pydicom, extrae el pixel_array y lo prepara para el preprocesamiento.

   Preprocess_img.py Realiza:
   1. Redimensionamiento (512x512)
   2. Conversión a escala de grises
   3. Ecualización de histograma (CLAHE)
   4. Normalización (0–1)
   5. Conversión a tensor (batch)

2. *Load_model.py:* Carga el modelo CNN entrenado desde el archivo WilhemNet86.h5.

3. *grad_cam.py:* Genera un mapa de calor utilizando Grad-CAM para visualizar las regiones importantes en la clasificación.
   

5. *Main.py:* Coordina todo el flujo:


   1.Recibe la imagen
   2. Ejecuta preprocesamiento
   3. Obtiene predicción
   4.Genera Grad-CAM
   5.Retorna resultados a la interfaz


<img width="613" height="880" alt="image" src="https://github.com/user-attachments/assets/3d6bc2bd-c455-4e97-95e7-c5b739406eff" />

