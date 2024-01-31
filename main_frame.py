# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:10:24 2024

@author: Digi_2
"""
import os
import subprocess
import sys
import traceback

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PyPDF2 import PdfReader
# import fitz
# from PIL import Image, ImageTk

#CloudCompare Python Plugin
import cccorelib
import pycc

#%% ADDING THE MAIN MODULE FOR ADDITIONAL FUNCTIONS
script_directory = os.path.abspath(__file__)
path_parts = script_directory.split(os.path.sep)

additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\main_module'
sys.path.insert(0, additional_modules_directory)

from main_gui import definition_of_labels_type_1

#%% ADDING GUIS

additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\segmentation_methods'
sys.path.insert(0, additional_modules_directory)

from supervised_machine_learning import GUI_mls
from unsupervised_machine_learning import GUI_mlu
from deep_learning_segmentation_v2 import GUI_dl

# additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\geometric-based_methods'
# sys.path.insert(0, additional_modules_directory)

# from supervised_geometric_based import GUI_gbs
# from unsupervised_geometric_based import GUI_gbu

# additional_modules_directory=os.path.sep.join(path_parts[:-1])+ '\radiometric-based_methods'
# sys.path.insert(0, additional_modules_directory)

# from radiometric_based import GUI_rb

#%% ADDING PDF FILE
current_directory= os.path.dirname(os.path.abspath(__file__))
path_pdf_file= os.path.join(current_directory,"help.pdf")


#%% GUI

class main_GUI(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        
        # Initialize the frames as attributes of the GUI class
        self.algo_mls_frame = GUI_mls(self, borderwidth=2, relief="groove", padx=10, pady=10)
        self.algo_mlu_frame = GUI_mlu(self, borderwidth=2, relief="groove", padx=10, pady=10)
        self.algo_dl_frame = GUI_dl(self, borderwidth=2, relief="groove", padx=10, pady=10)
        
        # self.algo_gbs_frame = GUI_gbs(self, borderwidth=2, relief="groove", padx=10, pady=10)
        # self.algo_gbu_frame = GUI_gbu(self, borderwidth=2, relief="groove", padx=10, pady=10)
        # self.algo_rb_frame = GUI_rb(self, borderwidth=2, relief="groove", padx=10, pady=10)
        
        self.algo_mls_frame.grid(row=3, rowspan=7, column=0, columnspan=3, pady=10, sticky="nsew")
        self.algo_mlu_frame.grid(row=2, column=0, pady=10, sticky="nsew")
        self.algo_dl_frame.grid(row=1, column=0, pady=10, sticky="nsew")
        
        # self.algo_gbs_frame.grid(row=3, column=0, pady=10, sticky="nsew")
        # self.algo_gbu_frame.grid(row=3, column=0, pady=10, sticky="nsew")
        # self.algo_rb_frame.grid(row=3, column=0, pady=10, sticky="nsew")
       
        # At the beginning, hide the frames of the algorithms
        self.algo_mls_frame.grid_forget()
        self.algo_mlu_frame.grid_forget()
        self.algo_dl_frame.grid_forget()
        
        # self.algo_gbs_frame.grid_forget()
        # self.algo_gbu_frame.grid_forget()
        # self.algo_rb_frame.grid_forget()
        
        # Agregar self.text y self.scrollbar como atributos
        self.text = tk.Text(self, wrap="none", height=20, width=60)
        self.text.grid(row=1, column=2, sticky="nsew")

        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.scrollbar.grid(row=1, column=3, sticky="ns")
        
        
    def hide_frames(self):
        # Hide all algorithm frames
        self.algo_mls_frame.grid_forget()
        self.algo_mlu_frame.grid_forget()
        self.algo_dl_frame.grid_forget()
        # self.algo_gbs_frame.grid_forget()
        # self.algo_gbu_frame.grid_forget()
        # self.algo_rb_frame.grid_forget()

    def toggle_mls(self, root_gui):
        # Show or hide widgets related to Supervised Machine Learning 
        self.hide_frames()
        # self.algo_mls_frame.show_frame(root_gui)
        self.algo_mls_frame.show_frame(root_gui)
        # self.algo_mls_frame.grid(row=3, column=0, pady=10, sticky="nsew")
        # self.algo_mls_frame.pack(side="left", fill="both", expand=True)
    
    def toggle_mlu(self, root_gui):
        # Show or hide widgets related to Unsupervised Machine Learning
        self.hide_frames()
        self.algo_mlu_frame.show_frame(root_gui)

    def toggle_dl(self, root_gui):
        # Show or hide widgets related to Deep Learning
        self.hide_frames()
        self.algo_dl_frame.show_frame(root_gui)

    def toggle_gbs(self, root_gui):
        # Show or hide widgets related to Supervised Geometric Based methods
        self.hide_frames()
        self.algo_gbs_frame.show_frame(root_gui)
        
    def toggle_gbu(self, root_gui):
        # Show or hide widgets related to Unsupervised Geometric Based methods
        self.hide_frames()
        self.algo_gbu_frame.show_frame(root_gui)

    def toggle_rb(self, root_gui):
        # Show or hide widgets related to Radiometric Based methods
        self.hide_frames()
        self.algo_rb_frame.show_frame(root_gui)
    
    def on_scroll(self, *args):
        # Actualizar la posición de la barra de desplazamiento en función de la posición de la barra de desplazamiento
        self.text.yview(*args)
        
    def open_pdf(self):
        try:
            # Intentar abrir el archivo PDF en la ruta predeterminada
            pdf_path = path_pdf_file
            pdf = PdfReader(pdf_path)

            # Mostrar cada página del PDF en el widget de texto
            for page_num in range(len(pdf.pages)):
                text_content = pdf.pages[page_num].extract_text()
                self.text.insert(tk.END, text_content)

            # Configurar la barra de desplazamiento
            self.text.config(yscrollcommand=self.scrollbar.set)
            self.scrollbar.config(command=self.on_scroll)

        except FileNotFoundError:
            # Si el archivo no se encuentra, mostrar un mensaje de error
            messagebox.showerror("Error", f"No se pudo encontrar el archivo PDF en la ruta predeterminada.")
        except Exception as e:
            # Para cualquier otro error, mostrar un mensaje de error
            messagebox.showerror("Error", f"No se pudo abrir el archivo PDF: {e}")
                
                
    def main_frame_gui(self, root_gui):
                
        # GENERAL CONFIGURATION OF THE GUI
        
        # Configuration of the window        
        root_gui.title("SEG 2 DIAGNOSE")
        root_gui.resizable(False, False)     
        root_gui.attributes('-toolwindow', -1)  # Remove minimize and maximize button 
        
        # Configuration of the tabs
        tab_control = ttk.Notebook(root_gui)
        tab_control.pack(expand=1, fill="both")
        
        tab1 = ttk.Frame(tab_control)  # Create 3 tabs
        tab1.pack()
        tab_control.add(tab1, text='Construction Systems Segmentation')
        tab2 = ttk.Frame(tab_control)  # Create 3 tabs
        tab2.pack() 
        tab_control.add(tab2, text='Damage evaluation')
        tab3 = ttk.Frame(tab_control)  # Create 3 tabs
        tab3.pack()
        tab_control.add(tab3, text='Other')
        tab4 = ttk.Frame(tab_control)  # Create 3 tabs
        tab4.pack()
        tab_control.add(tab4, text='About')
        tab_control.pack(expand=1, fill="both")
        
        # TAB1 = CONSTRUCTION SYSTEMS SEGMENTATION
        
        # Labels
        label_texts = [
            "Machine Learning",
            "Deep Learning"
        ]
        row_positions = [0,1]        
        definition_of_labels_type_1 ("t1",label_texts,row_positions,tab1,0)
        
        label_texts = [
            "For more information:"
        ]
        row_positions = [0]        
        definition_of_labels_type_1 ("t1",label_texts,row_positions,tab1,4)

        
        # Buttons
        button_mls = ttk.Button(tab1, text="Supervised", command=lambda: self.toggle_mls(root_gui))
        button_mlu = ttk.Button(tab1, text="Unsupervised", command=lambda: self.toggle_mlu(root_gui))
        button_dl = ttk.Button(tab1, text="Deep Learning", command=lambda: self.toggle_dl(root_gui))
        button_mls.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        button_mlu.grid(row=0, column=2, padx=10, sticky="w")
        button_dl.grid(row=1, column=1, padx=10, sticky="w")
        
        # Create a vertical line (separator) in the middle of the window
        separator = ttk.Separator(tab1, orient="vertical")
        separator.grid(row=0, column=3, rowspan=5, sticky="ns", padx=5)
        
        # # Create a PDF visualizer in the right side of the window
        # text_frame = tk.Frame(tab1, padx=5, pady=5)
        # self.text = tk.Text(text_frame, wrap="word", height=20, width=60)
        # scrollbar = ttk.Scrollbar(text_frame, command=self.text.yview)
        # self.text.config(yscrollcommand=scrollbar.set)
        # self.text.grid(row=1, column=2, sticky="nsew")
        # scrollbar.grid(row=1, column=3, sticky="ns")
        # text_frame.grid(row=1, column=4, rowspan=10, padx=5, sticky="nsew")
        
        # Create a text viewer in the right side of the window
        self.text = tk.Text(tab1, wrap="none", height=20, width=60)
        self.text.grid(row=1, column=4, rowspan=10, padx=5, sticky="nsew")

        # Crear una barra de desplazamiento vertical
        self.scrollbar = tk.Scrollbar(tab1, orient="vertical", command=self.text.yview)
        self.scrollbar.grid(row=1, column=5, rowspan=10, sticky="ns")
        
        self.open_pdf()
        
        # # Crear el botón para abrir un PDF
        # button_open_pdf = ttk.Button(tab1, text="Abrir PDF", command = self.open_pdf(root_gui))
        # button_open_pdf.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        # Configurar el redimensionamiento de las columnas y filas del Frame principal
        # root_gui.columnconfigure(2, weight=1)
        # root_gui.rowconfigure(0, weight=1)
        
        
        # TAB2 = DAMAGE EVALUATION
        
        # Labels
        label_texts = [
            "Machine Learning",
            "Deep Learning"
            "Features Computation",
            "Arches and Vaults",
            "Slabs",
            "Pilars and Buttresses"
            "BIM integration"
        ]
        row_positions = [0,1,2,3,4,5,6]        
        definition_of_labels_type_1 ("t2",label_texts,row_positions,tab2,0)
        
        # Crear botones como flechas para mostrar/ocultar el contenido
        button_gbs = ttk.Button(tab2, text="Supervised", command=lambda: self.toggle_gbs(root_gui))
        button_gbu = ttk.Button(tab2, text="Unsupervised", command=lambda: self.toggle_gbu(root_gui))
        button_rb = ttk.Button(tab2, text="Radiometric based", command=lambda: self.toggle_rb(root_gui))
        button_gbs.grid(row=0, column=1, padx=5, sticky="w")
        button_gbu.grid(row=0, column=2, padx=5, sticky="w")
        button_rb.grid(row=1, column=1, padx=5, sticky="w")
        
        
        # Create a vertical line (separator) in the middle of the window
        separator = ttk.Separator(tab2, orient="vertical")
        separator.grid(row=0, column=3, rowspan=5, sticky="ns", padx=5)
        
        self.text = tk.Text(tab2, wrap="none", height=20, width=60)
        self.text.grid(row=1, column=4, rowspan=10, padx=5, sticky="nsew")

        # Crear una barra de desplazamiento vertical
        self.scrollbar = tk.Scrollbar(tab2, orient="vertical", command=self.text.yview)
        self.scrollbar.grid(row=1, column=5, rowspan=10, sticky="ns")
        
        self.open_pdf()
        
        
        
        # TAB3 = OTHER
        
        # Labels
        label_texts = [
            "Noise Reduction",
            "Point Cloud Voxelization",
            "Potree Converter"
        ]
        row_positions = [0,1,2]        
        definition_of_labels_type_1 ("t3",label_texts,row_positions,tab3,0)
        
        button_nr = ttk.Button(tab2, text="Noise Reduction", command=lambda: self.toggle_nr(root_gui))
        button_pcv = ttk.Button(tab2, text="Point Cloud Voxelization", command=lambda: self.toggle_pcv(root_gui))
        button_pc = ttk.Button(tab2, text="Potree Converter", command=lambda: self.toggle_pc(root_gui))
        button_nr.grid(row=0, column=1, padx=5, sticky="w")
        button_pcv.grid(row=1, column=1, padx=5, sticky="w")
        button_pc.grid(row=2, column=1, padx=5, sticky="w")

        # TAB4 = ABOUT
        
try:
    # START THE MAIN WINDOW        
    root_gui = tk.Tk()
    app = main_GUI()
    app.main_frame_gui(root_gui)
    root_gui.mainloop()    
except Exception as e:
    print("An error occurred during the computation of the algorithm:", e)
    # Optionally, print detailed traceback
    traceback.print_exc()
    root_gui.destroy()







# def toggle_ml(root_gui):
#     # Mostrar u ocultar widgets relacionados con Machine Learning
#     if algo_ml_frame.winfo_ismapped(root_gui):
#         algo_ml_frame.hide_frame(root_gui)
#     else:
#         algo_ml_frame.show_frame(root_gui)
#         algo_dl_frame.hide_frame(root_gui)

# def toggle_dl(root_gui):
#     # Mostrar u ocultar widgets relacionados con Deep Learning
#     if algo_dl_frame.winfo_ismapped(root_gui):
#         algo_dl_frame.hide_frame(root_gui)
#     else:
#         algo_dl_frame.show_frame(root_gui)
#         algo_ml_frame.hide_frame(root_gui)

# # Crear la ventana principal
# root_gui = tk.Tk()
# root_gui.title("Selección de Algoritmo")

# # Crear el Frame principal
# frame_main = tk.Frame(root_gui)
# frame_main.grid(padx=10, pady=10)

# # Crear botones como flechas para mostrar/ocultar el contenido
# button_ml = ttk.Button(frame_main, text="↓ Machine Learning", command=toggle_ml)
# button_dl = ttk.Button(frame_main, text="↓ Deep Learning", command=toggle_dl)

# # Colocar los botones en la parte izquierda de la ventana
# button_ml.grid(row=0, column=0, padx=5, sticky="w")
# button_dl.grid(row=0, column=0, padx=5, sticky="w")

# # Crear una línea vertical (Separator) en la mitad de la ventana
# separator = ttk.Separator(frame_main, orient="vertical")
# separator.grid(row=0, column=1, rowspan=2, sticky="ns", padx=5)

# # Crear el Frame para los algoritmos de Machine Learning
# algo_ml_frame = GUI2(frame_main, borderwidth=2, relief="groove", padx=10, pady=10)

# # Crear el Frame para los algoritmos de Deep Learning
# algo_dl_frame = GUI2(frame_main, borderwidth=2, relief="groove", padx=10, pady=10)

# # Crear un visor de PDF en la parte derecha de la ventana
# text_frame = tk.Frame(frame_main, padx=10, pady=10)
# text = tk.Text(text_frame, wrap="word", height=20, width=60)
# scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
# text.config(yscrollcommand=scrollbar.set)
# text.grid(row=0, column=0, sticky="nsew")
# scrollbar.grid(row=0, column=1, sticky="ns")
# text_frame.grid(row=0, column=2, rowspan=2, padx=5, sticky="nsew")

# # Crear el botón para abrir un PDF
# # button_open_pdf = ttk.Button(frame_main, text="Abrir PDF", command=open_pdf)
# # button_open_pdf.grid(row=1, column=0, padx=5, pady=5, sticky="w")

# # Configurar el redimensionamiento de las columnas y filas del Frame principal
# frame_main.columnconfigure(2, weight=1)
# frame_main.rowconfigure(0, weight=1)

# # Iniciar el bucle principal
# root_gui.mainloop()




# class MainFrame(tk.Frame):
#     def __init__(self, master=None, **kwargs):
#         super().__init__(master, **kwargs)

#         # Crear botones como flechas para mostrar/ocultar el contenido
#         self.button_ml = ttk.Button(self, text="↓ Machine Learning", command=self.toggle_ml)
#         self.button_dl = ttk.Button(self, text="↓ Deep Learning", command=self.toggle_dl)

#         # Crear el Frame para los algoritmos de Machine Learning
#         self.algo_ml_frame = GUI2(self, borderwidth=2, relief="groove", padx=10, pady=10)
#         self.algo_ml_frame.grid(row=1, column=0, pady=10, sticky="nsew")

#         # Crear el Frame para los algoritmos de Deep Learning
#         self.algo_dl_frame = GUI2(self, borderwidth=2, relief="groove", padx=10, pady=10)
#         self.algo_dl_frame.grid(row=1, column=0, pady=10, sticky="nsew")

#         # Configurar el redimensionamiento de las columnas y filas
#         self.columnconfigure(0, weight=1)
#         self.rowconfigure(1, weight=1)

#         # Inicialmente, ocultar los frames de algoritmos
#         self.algo_ml_frame.grid_forget()
#         self.algo_dl_frame.grid_forget()
        
#     def toggle_ml(self):
#         # Mostrar u ocultar widgets relacionados con Machine Learning
#         self.algo_ml_frame.grid(row=1, column=0, pady=10, sticky="nsew")
#         self.algo_dl_frame.pack_forget()

#     def toggle_dl(self):
#         # Mostrar u ocultar widgets relacionados con Deep Learning
#         self.algo_dl_frame.grid(row=1, column=0, pady=10, sticky="nsew")
#         self.algo_ml_frame.pack_forget()
        
# # def open_pdf():
# #     # Abrir un archivo PDF y cargarlo en el visor
# #     pdf_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
# #     if pdf_path:
# #         try:
# #             pdf = PdfReader(pdf_path)
# #             text.delete(1.0, tk.END)
# #             for page_num in range(len(pdf.pages)):
# #                 text.insert(tk.END, pdf.pages[page_num].extract_text())
# #         except Exception as e:
# #             messagebox.showerror("Error", f"No se pudo abrir el archivo PDF: {e}")


# def main():
#     # Crear la ventana principal
#     root_gui = tk.Tk()
#     root_gui.title("Selección de Algoritmo")

#     # Crear el Frame principal
#     frame_main = MainFrame(root_gui)
#     frame_main.grid(padx=10, pady=10, sticky="nsew")
    
#     # Crear un visor de PDF en la parte derecha de la ventana
#     text_frame = tk.Frame(frame_main, padx=10, pady=10)
#     text = tk.Text(text_frame, wrap="word", height=20, width=60)
#     scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
#     text.config(yscrollcommand=scrollbar.set)
#     text.grid(row=0, column=0, sticky="nsew")
#     scrollbar.grid(row=0, column=1, sticky="ns")
#     text_frame.grid(row=0, column=2, rowspan=2, padx=5, sticky="nsew")
    
#     # # Crear el botón para abrir un PDF
#     # button_open_pdf = ttk.Button(frame_main, text="Abrir PDF", command=open_pdf)
#     # button_open_pdf.grid(row=1, column=0, padx=5, pady=5, sticky="w")

#     # Configurar el redimensionamiento de las columnas y filas del Frame principal
#     frame_main.columnconfigure(0, weight=1)
#     frame_main.columnconfigure(2, weight=1)
#     frame_main.rowconfigure(0, weight=1)

#     # Iniciar el bucle principal
#     root_gui.mainloop()


# if __name__ == "__main__":
#     main()