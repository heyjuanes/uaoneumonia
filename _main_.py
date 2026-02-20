#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfaz gráfica (UI) para detección de neumonía.

Este archivo contiene únicamente la lógica de UI (Tkinter) y delega la
inferencia y Grad-CAM al integrador:
    src/app/integrator.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from tkinter import END, Tk, Text, StringVar
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from turtle import stamp

from PIL import ImageTk, Image
import tkcap

from src.app.integrator import PneumoniaDetector

from src.data.read_img import read_dicom_image, read_image_file

from datetime import datetime



class App:
    """
    Aplicación Tkinter para cargar imagen, ejecutar predicción y exportar PDF/CSV.
    """

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        # Detector (carga el modelo una sola vez)
        self.detector = PneumoniaDetector(model_path="models/conv_MLP_84.h5")
        self.filepath: str | None = None

        # Fuente negrita
        font_bold = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # Labels
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=font_bold)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=font_bold)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=font_bold)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=font_bold)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=font_bold,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=font_bold)

        # Variables
        self.ID = StringVar()

        # Input boxes
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        # Areas para imágenes y salida
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        # Botones
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        # Posiciones
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)

        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)

        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        # Focus en ID
        self.text1.focus_set()

        # Estado
        self.label: str | None = None
        self.proba: float | None = None
        self.heatmap = None
        self.report_id = 0

        # Ejecutar UI
        self.root.mainloop()

    def load_img_file(self) -> None:
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
     )
        if not filepath:
            return

        self.filepath = filepath

        # Leer SOLO para visualizar (sin inferencia)
        if filepath.lower().endswith(".dcm"):
         _, pil_img = read_dicom_image(filepath)
        else:
         _, pil_img = read_image_file(filepath)

        # Limpiar UI
        self._clear_image_box(self.text_img1)
        self._clear_image_box(self.text_img2)
        self._clear_text_outputs()

        # Mostrar imagen original
        self.img1 = pil_img.resize((250, 250), Image.Resampling.LANCZOS)
        self.img1 = ImageTk.PhotoImage(self.img1)
        self.text_img1.image_create(END, image=self.img1)

        # Habilitar predecir
        self.button1["state"] = "enabled"


    def run_model(self) -> None:
        """
        Ejecuta la inferencia y muestra resultado + probabilidad + heatmap.
        """
        if not self.filepath:
            showinfo(title="Error", message="Primero debes cargar una imagen.")
            return

        result = self.detector.predict(self.filepath)
        self.label = result.label
        self.proba = result.probability
        self.heatmap = result.heatmap

        # Mostrar heatmap (numpy -> PIL -> Tk)
        self.img2 = Image.fromarray(self.heatmap).resize(
            (250, 250), Image.Resampling.LANCZOS
        )
        self.img2 = ImageTk.PhotoImage(self.img2)

        self._clear_image_box(self.text_img2)
        self.text_img2.image_create(END, image=self.img2)

        # Mostrar texto
        self._clear_text_outputs()
        self.text2.insert(END, self.label)
        self.text3.insert(END, f"{self.proba:.2f}%")

    def save_results_csv(self) -> None:
        """
        Guarda ID, clase y probabilidad en historial.csv.
        """
        if self.label is None or self.proba is None:
            showinfo(title="Guardar", message="Primero debes ejecutar una predicción.")
            return

        with open("historial.csv", "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([self.text1.get(), self.label, f"{self.proba:.2f}%"])

        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self) -> None:
        """
        Captura la ventana actual y genera un PDF (nombre único por timestamp).
        """
        cap = tkcap.CAP(self.root)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jpg_name = f"Reporte_{stamp}.jpg"
        pdf_name = f"Reporte_{stamp}.pdf"

        cap.capture(jpg_name)

        img = Image.open(jpg_name).convert("RGB")
        img.save(pdf_name)

        showinfo(title="PDF", message=f"El PDF fue generado con éxito: {pdf_name}")


    def delete(self) -> None:
        """
        Limpia el formulario y las imágenes.
        """
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if not answer:
            return

        self.text1.delete(0, "end")
        self._clear_text_outputs()
        self._clear_image_box(self.text_img1)
        self._clear_image_box(self.text_img2)

        self.filepath = None
        self.label = None
        self.proba = None
        self.heatmap = None
        self.button1["state"] = "disabled"

        showinfo(title="Borrar", message="Los datos se borraron con éxito")

    @staticmethod
    def _clear_image_box(text_widget: Text) -> None:
        """
        Limpia un widget Text usado para mostrar imágenes.
        """
        text_widget.delete("1.0", "end")

    def _clear_text_outputs(self) -> None:
        """
        Limpia los campos de salida (resultado y probabilidad).
        """
        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")


def main() -> int:
    App()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
