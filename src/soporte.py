import pandas as pd
import re
import numpy as np
import pickle


class Extraccion():
    
    ''' 
    Clase para extraer datos de archivos csv, txt y xml y unirlos en un dataframe.
    
    Atributos:
    - diccionario de archivos como **kwargs: los keys son los nombres de los dataframes que queramos
    poner a cada archivo, y los values son los nombres de los archivos.
    
    Metodos:
    cargar_datos()
    Acepta una ruta y el nombre de archivos de csv, txt o xml y carga los datos de los archivos en dataframes
        guardados como values en un diccionario.
        
    unir_datos()
    Acepta un diccionario de dataframes como resultado de la funcion de cargar_datos y los une en un dataframe 
        por las columnas con un outer merge por las indices.
    '''
    
    def __init__(self, **diccionario_archivos):
                
        '''Construye los atributos del objeto de extraccion de datos de archivos.
        
        Parametros:
        - diccionario como **kwargs donde los nombres de los archivos son los values y los nombres que queremos
        poner a los respectivos dataframes son los keys 
        '''
        
        self.diccionario_archivos = diccionario_archivos
        
    def cargar_datos(self, ruta):
    
        '''
        Acepta una ruta y el nombre de archivos de csv, txt o xml y carga los datos de los archivos en dataframes
        guardados como values en un diccionario.
        
        Parametros:
        - string: la ruta relativa de los archivos
        - self.diccionario_archivos 
        
        Returns:
        - diccionario de dataframes
        '''
        
        diccionario_df = {}
        
        for k, v in self.diccionario_archivos.items():
            if 'csv' in v:
                diccionario_df[k] = pd.read_csv(f"{ruta}{v}", index_col=0)
            elif 'txt' in v:
                sep = input(f'Escribe el separador para el archivo txt {v}')
                diccionario_df[k] = pd.read_csv(f"{ruta}{v}", sep = sep, index_col=0)
            elif 'xml' in v:
                diccionario_df[k] = pd.read_xml(f"{ruta}{v}")
        
        self.diccionario_df = diccionario_df
        
        return diccionario_df
    
    
    def unir_datos(self):
    
        '''
        Acepta un diccionario de dataframes como resultado de la funcion de cargar_datos y los une en un dataframe 
        por las columnas con un outer merge por las indices.
        
        Parametros:
        - self.diccionario_df: un diccionario de dataframes
        
        Returns:
        - dataframe de los datos unidos
        '''
        
        lista_df = list(self.diccionario_archivos.keys())
        
        df = self.diccionario_df[lista_df[0]].merge(self.diccionario_df[lista_df[1]], on = "index", how = "outer")
        
        for i in range(1, len(lista_df)-1):
            df = df.merge(self.diccionario_df[lista_df[i+1]], on = "index", how = "outer")
        
        return df
    
    
class Limpieza:
    
    def __init__(self, dataframe):
        '''Construye los atributos del objeto de limpieza de un dataframe.
    
        Parametros:
        - dataframe con datos para limpiar 
        '''
        
        self.df = dataframe
        
    def limpiar_espacios(self, columna):
        
        '''
        Acepta una columna de tipo de datos objeto de un dataframe y si el valor tiene espacios delante de 
        comas, devuelve el valor con los espacios despues de las comas.
        
        Parametros:
        - valores de columna de un dataframe de tipo objeto
        
        Returns:
        - valor de la columna del dataframe limpiado o un np.nan si el valor es nulo
        '''
        try:
            patron_espacios = "\s*,\S+"
            return re.sub(patron_espacios, ", ", columna)

        except:
            return np.nan
    
    def separar_comas(self, columna):
        
        '''
        Acepta una columna de tipo de datos objeto de un dataframe y si el valor tiene comas delante de un espacio
        y una 'a' minuscula, reemplaza la coma por punto y coma (';').
        
        Parametros:
        - valores de columna de un dataframe de tipo objeto
        
        Returns:
        - valor de la columna del dataframe con comas delantes de 'a's minusculas reemplazados por ';'
        '''
        
        try:
            patron_comas = '(,\s)a'
            return re.sub(patron_comas, "; a", columna)
        except:
            return np.nan
        
    