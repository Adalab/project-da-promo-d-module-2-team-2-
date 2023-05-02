import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Extraccion:
    
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
        Para usar con un apply. Acepta una columna de tipo de datos objeto de un dataframe y si el valor tiene espacios delante de 
        comas, devuelve el valor con los espacios despues de las comas.
        
        Parametros:
        - string: valores de columna de un dataframe de tipo objeto
        
        Returns:
        - valor de la columna del dataframe limpiado o un np.nan si el valor es nulo
        '''
        try:
            patron_espacios = "\s*,\S+"
            return re.sub(patron_espacios, ", ", columna)

        except:
            return np.nan
    
    def comas_as(self, columna):
        
        '''
        Para usar con un apply. Acepta una columna de tipo de datos objeto de un dataframe y si el valor tiene comas delante de un espacio
        y una 'a' minuscula, reemplaza la coma por punto y coma (';').
        
        Parametros:
        - string: valores de columna de un dataframe de tipo objeto
        
        Returns:
        - valor de la columna del dataframe con comas delantes de 'a's minusculas reemplazados por ';'
        '''
        
        try:
            patron_comas = '(,\s)a'
            return re.sub(patron_comas, "; a", columna)
        except:
            return np.nan
        
    def comas_parentesis(self, columna):
    
        '''
        Para usar con un apply. Acepta una columna de tipo de datos objeto de un dataframe y si el valor tiene parentesis, 
        reemplaza las comas entre parentesis por punto y coma (';').
        
        Parametros:
        - string: valores de columna de un dataframe de tipo objeto
        
        Returns:
        - valor de la columna del dataframe con comas dentro de parentesis reemplazados por ';'
        - np.nan si el valor es nulo
        '''
        
        try:
            patron = r'(\(.*),(\s.*),(\s.*\))'
            return re.sub(patron, r'\1;\2;\3', columna)
        
        except:
            return np.nan
        
    
    def eliminar_columnas (self, dataframe, *lista_columnas):
        
        '''
        Acepta un dataframe y una lista de columnas y elimina las columnas especificadas del dataframe. 
        Muestra una lista de las columnas que quedan en el el dataframe despues de ejecutar la funcion.

        Parametros:
        - dataframe
        - lista de nombres de columnas

        Returns:
        - none
        '''

        for columna in dataframe.columns:
            for elemento in lista_columnas:
                if elemento == columna:
                    #esta función itera por nuestra lista_columna (el arg) y comprueba si el elemento (la etiqueta a eliminar)
                    # se encuentra en cada columna, y, de ser así, la eliminará usando el método .drop()
                    dataframe.drop([columna],axis = 1, inplace = True)
        
        print(f"Se han eliminado las columnas indicadas. Ahora las columnas del dataframe son {dataframe.columns}")
    
    def eliminar_rango_col(self, dataframe, columna_inicio, columna_final):
    
        '''
        Acepta un dataframe y los nombres de dos columnas, una del inicio y la otra del final de un rango de columnas, 
        entre lo cual se eliminaran todas las columnas, las del inicio y final incluidas. Muestra una lista de las columnas 
        que quedan en el el dataframe despues de ejecutar la funcion.
        
        Parametros:
        - dataframe
        - string: nombre de la columna donde inicia el rango de columnas que eliminar
        - string: nombre de la columna donde finaliza el rango de columnas que eliminar
        
        Returns:
        -none
        '''
        #creamos un diccionario de los índices por columna de nuestro df
        diccionario = {}
        
        #los nombres de las columnas serán nuestras keys para poder acceder a los índices usando los nombres de las columnas (los parámetros)
        for indice, col in enumerate(dataframe.columns):
            diccionario.update({col: indice})
                
        #usamos el diccionario para obtener los índices en el rango 
        if columna_inicio in diccionario:
            indice1 = diccionario.get(columna_inicio)
        else: 
            print('Columna1 no está en el diccionario')
        
        if columna_final in diccionario:
            #usamos el +1 para poder acceder al último índice
            indice2 = diccionario.get(columna_final) + 1
        else: 
            print('Columna2 no está en el diccionario')
            
        #usamos los índices para borrar el rango de columnas y así eliminiar varias a la vez
        for elemento in dataframe.columns[indice1:indice2]:    
            dataframe.drop([elemento],axis = 1, inplace = True)
            
        print(f"Se han eliminado las columnas indicadas. Ahora las columnas del dataframe son: {dataframe.columns}")
        
    def col_categorias(self, dataframe, **diccionario_columnas):
        
        ''' 
        Acepta un dataframe y un diccionario del cual las keys son nombres de columnas del dataframe y los values
        son diccionarios que se usarán para mapear los valores de la columna a categorías en la columna
        especificada por el usuario con un input (puede ser una columna nueva o sobreescribir la columna).
        
        ParametrosÑ
        - dataframe
        - diccionario del cual las keys son nombres de columnas del dataframe y los values
        son diccionarios que se usarán para mapear los valores de la columna a categorías
        
        Returns
        - none
        '''
        for k, v in diccionario_columnas.items():
            dataframe[input(f'Escribe la nombre de la columna de categorías de {k}')] = dataframe[k].map(v, na_action='ignore')
            
    def juntar_columnas(self, dataframe, columna_inicio, columna_final, nueva_columna):
    
        '''
        Acepta un dataframe y los nombres de dos columnas, una del inicio y la otra del final de un rango de columnas, 
        y el nombre de una nueva columna. Entre el rango especificado, se juntaran los valores de las columnas en una lista 
        en una nueva columna con el nombre indicado. 
        
        Parametros:
        - dataframe
        - string: nombre de la columna del inicio del rango
        - string: nombre de la columna del final del rango
        - string: nombre de la columna nueva donde se guardaran los valores juntados
        
        Returns:
        - none    
        '''
        
        diccionario = {}

        for i, f in dataframe.loc[:, columna_inicio:columna_final].iterrows():
            diccionario[i] = []
            
            for e in f:
                if type(e) == str:
                    diccionario[i].append(e)
        
        dataframe[nueva_columna] = diccionario.values()
        dataframe[nueva_columna] = dataframe[nueva_columna].apply(lambda y: np.nan if len(y)==0 else ','.join(y))
    
    def unos_zeros(self, columna, string):
        
        ''' 
        Para usar con un apply. Acepta una columna de tipo objeto y un string. Si el string aparece en el
        valor de la columna, el valor se convierte en un 1.0, si no se encuentra el valor, el valor se convierte
        en 0. Un np.nan se mantiene como np.nan.
        
        Parametros:
        - valores de la columna de un dataframe
        - string
        
        Returns:
        - 1.0 si el string aparece en el valor de la columna
        - 0.0 si el string no aparece en en valor de la columna
        - np.nan si el valor es np.nan        
        '''
        
        try:
            if string in columna:
                return 1
            else:
                return 0
        except:
            return np.nan
        
    
    def dividir_columnas(self, dataframe, lista_columnas, **dicc_respuestas):
        ''' 
        Acepta un dataframe, una lista de columnas, y un diccionario del cual las keys son los nombres de
        las columnas del dataframe y los valores son los valores unicos de la columna. Para cada columna en la 
        lista se crea una columna nueva en el dataframe de cada valor unico de la columna. En cada columna nueva 
        los valores son unos o zeros segun se el respectivo valor único aparece en el valor de la columna original o
        
        Parametros:
        - dataframe
        - lista de columnas
        - diccionario como **kwargs del cual las keys son los nombres de las columnas del dataframe y los valores son 
        los valores unicos de la columna
        
        Returns:
        -none
        '''
        
        for col in lista_columnas:
            lista_nombres = []
            lista_respuestas = dicc_respuestas[col]
            for elemento in lista_respuestas:
                nombre_columna = f"{col}_{elemento.strip()}"
                lista_nombres.append(nombre_columna)
                dataframe[nombre_columna] = dataframe.apply(lambda df: self.unos_zeros(df[col], elemento), axis=1)
        
    
      
    def unos_valores(self, dataframe, *lista_columnas):
        '''
        Acepta un dataframe y una lista de prefijos de nombres de columnas como *args. Para cada columna en la lista,
        reemplaza los unos por el el valor correspondiente a la columna en string.
        
        Parametros:
        - dataframe
        - lista de nombres de columnas
        
        Returns:
        - none    
        '''
        
        for col in dataframe.columns:
            for i in lista_columnas:
                if i in col:
                    dataframe[col] = dataframe[col].astype('object')
                    dataframe[col] = dataframe[col].apply(lambda x: col.replace(i, '') if x == 1.0 else x)
        
    
    
    
    
class Exploracion:
        
    def __init__(self, dataframe):
        '''Construye los atributos del objeto de exploracion de datos en dataframes.
        
        Parametros:
        - dataframe
        '''    
            
        self.dataframe = dataframe
    
    def explore(self, dataframe, nombre):
        '''
        Acepta un dataframe y el nombre o titulo del dataframe como string, y imprime informacion
        sobre el dataframe: el numero de filas y columnas, las columnas, una muestra de dos filas,
        los principales estadísticos de las variables categóricas, y el porcentaje de nulos por columna.
        
        Parametros:
        - dataframe
        - string: nombre o titulo del dataframe
        
        Returns:
        - none
        '''
        print(f"EXPLORACIÓN DEL DATAFRAME {nombre.upper()}")
        print("--------------")
        print(f"El dataframe {nombre} tiene {dataframe.shape[0]} filas y {dataframe.shape[1]} columnas")
        print("--------------")
        print(f"El dataframe {nombre} tiene las siguientes columnas:")
        for col in dataframe.columns:
            print(col)     
        print("--------------")
        print("Una muestra de dos filas seleccionadas al azar:")
        display(dataframe.sample(2))
        print("--------------")
        print(f"Los principales estadísticos de las variables categóricas son:")
        display(dataframe.describe(include=object).T)
        print("--------------")
        print("El porcentaje de nulos por columna:")
        for i, col in enumerate(dataframe.isnull().sum()):
            print(f"{dataframe.isnull().sum().index[i]}: {col/dataframe.shape[0]*100}")
        print("--------------")
    
                
    def dict_respuestas(self, dataframe):
    
        ''' 
        Acepta un dataframe y crea un diccionario con los nombres de las columnas como keys 
        y una lista de los valores únicos de la columna como values.
        
        Parametros:
        - dataframe
        
        Returns:
        - diccionario con los nombres de las columnas del dataframe como keys 
        y una lista de los valores únicos de la columna como values

        '''
        dic_respuestas = {}
        
        for col in self.dataframe.columns:
            lista_nueva = []
            #ignoramos la columna index
            if col == "index":
                pass
            #comprobamos si los valores únicos son más de 20 - si es una pregunta multirespuesta 
            elif len(self.dataframe[col].unique()) > 22:
                #crea una lista de los valores únicos
                    lista_unicos = list(self.dataframe[col].unique())
                    for sublist in lista_unicos:
                        #cada lista dentro de los valores únicos se divide por las comas 
                        try:
                            lista_nueva.extend(sublist.split(","))
                        except:
                            pass
                        #lista de todas las posibles respuestas individuales
                        #la convertimos en set para eliminar los duplicados 
                        set_unicos = set(lista_nueva)
                        #lo volvemos a convertir en lista para poder usarla cómodamente
                        lista_nueva = list(set_unicos)
            else:
                #si no es multirespuesta podemos usar el unique para ver los valores únicos 
                lista_nueva = list(self.dataframe[col].unique())
            #list comprehension para quitar los nulos    
            lista_sin_nan = [item for item in lista_nueva if not (pd.isnull(item)) == True]
            #metemos la lista sin nulos en un diccionario donde el key es el número de la columna y los values con la lista de valores únicos lista_sin_nan
            dic_respuestas[col] = lista_sin_nan

        for k, v in dic_respuestas.items():
            try:
                for i, x in enumerate(v):
                    dic_respuestas[k][i] = x.strip()
            except:
                pass
            
        return dic_respuestas    
    
    def count_respuestas(self, dataframe, **diccionario):
    
        '''
        Acepta un dataframe y un diccionario del cual las keys son los nombres de
        las columnas del dataframe y los valores son los valores unicos de la columna. 
        Para cada columna imprime el porcentaje de respuestas que compone cada valor unico.
        
        Parametros:
        - dataframe
        - diccionario de respuestas
        
        Returns:
        - none
        '''
        
        for key in diccionario:
            #usamos el diccionario, sacamos los valores únicos de cada pregunta para encontrar la columna del dataframe, 
            # y el sum nos dice cuántas veces aparece      
            if key == "index":
                pass
            elif len(dataframe[key].unique()) > 22:
                print("----------------")
                print(f"PREGUNTA MULTIRESPUESTA {key}")  
                for respuesta in diccionario[key]:
                    try:
                        total = dataframe[key].str.contains(respuesta, regex = True).sum()
                        print(f"{respuesta}: total: {total}, {round(total/dataframe.shape[0]*100, 1)}%")
                    except:
                        print(f"no se puede para {respuesta}")
            else:
                print("------------")
                print(f"PREGUNTA {key} (%)")
                print(f"{round((dataframe[key].value_counts()/dataframe.shape[0]*100),2)}")

