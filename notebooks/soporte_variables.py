

diccionario_nombres = {'age':'age',
                        'gender':'gender',
                        'Q3':'country',
                        'Q5':'job_title',
                        'Q6':'years_programming',
                        'Q7':'dev_language',
                        'Q8':'first_language_rec',
                        'Q9':'IDE',
                        'Q10_Part_1':'notebooks_KaggleNotebooks',
                        'Q10_Part_2':'notebooks_ColabNotebooks',
                        'Q10_Part_3':'notebooks_AzureNotebooks',
                        'Q10_Part_4':'notebooks_Paperspace/Gradient',
                        'Q10_Part_5':'notebooks_Binder/JupyterHub',
                        'Q10_Part_6':'notebooks_CodeOcean',
                        'Q10_Part_7':'notebooks_IBMWatsonStudio',
                        'Q10_Part_8':'notebooks_AmazonSagemakerStudioNotebooks',
                        'Q10_Part_9':'notebooks_AmazonEMRNotebooks',
                        'Q10_Part_10':'notebooks_GoogleCloudNotebooks(AIPlatform/VertexAI)',
                        'Q10_Part_11':'notebooks_GoogleCloudDatalab',
                        'Q10_Part_12':'notebooks_DatabricksCollaborativeNotebooks',
                        'Q10_Part_13':'notebooks_Zeppelin/ZeplNotebooks',
                        'Q10_Part_14':'notebooks_DeepnoteNotebooks',
                        'Q10_Part_15':'notebooks_ObservableNotebooks',
                        'Q10_Part_16':'notebooks_None',
                        'Q10_OTHER':'notebooks_Other',
                        'Q14':'visualisation',
                        'Q41':'primary_data_tool',
                        'Q15':'ML',
                        'Q22':'size_DA_dept',
                        'Q24':'work_activities',
                        'Q32':'big_data',
                        'Q34':'BI_tools',
                        'Q39_Part_1':'sharing_PlotlyDash',
                        'Q39_Part_2':'sharing_Streamlit',
                        'Q39_Part_3':'sharing_NBViewer',
                        'Q39_Part_4':'sharing_GitHub',
                        'Q39_Part_5':'sharing_Personalblog',
                        'Q39_Part_6':'sharing_Kaggle',
                        'Q39_Part_7':'sharing_Colab',
                        'Q39_Part_8':'sharing_Shiny',
                        'Q39_Part_9':'sharing_does_not_share',
                        'Q39_OTHER':'sharing_Other'}

lista_nombres_nuevos = ['age', 
                        'gender', 
                        'country', 
                        'job_title', 
                        'years_programming', 
                        'dev_language', 
                        'first_language_rec', 
                        'IDE', 
                        'notebooks', 
                        'visualisation', 
                        'ML', 
                        'size_DA_dept', 
                        'work_activities', 
                        'big_data', 
                        'BI_tools', 
                        'sharing']

dicc_continentes = {'Europe': ['Greece', 'Belgium', 'Poland', 'Italy', 'Spain', 'United Kingdom of Great Britain and Northern Ireland', 'France',
                            'Switzerland', 'Sweden', 'Netherlands', 'Ukraine', 'Romania', 'Austria', 'Belarus', 'Ireland',
                            'Portugal', 'Denmark', 'Germany', 'Norway', 'Czech Republic'],
                    'Asia & Oceania': ['Australia', 'India', 'Indonesia', 'Pakistan', 'Russia', 'Turkey', 'Japan', 'Singapore', 'China', 'Iran, Islamic Republic of...', 
                             'Viet Nam', 'Israel', 'Bangladesh', 'Saudi Arabia', 'Taiwan', 'Hong Kong (S.A.R.)', 'South Korea', 'Philippines', 'Sri Lanka', 
                             'United Arab Emirates', 'Malaysia', 'Thailand', 'Nepal', 'Kazakhstan', 'Iraq'],
                    'America': ['Mexico', 'Brazil', 'United States of America',
                                'Peru', 'Argentina', 'Colombia', 'Canada', 'Chile', 'Ecuador'],
                    'Africa':['Uganda', 'Ghana','Algeria', 'Tunisia', 'South Africa', 'Nigeria', 'Kenya','Egypt', 'Ethiopia', 'Morocco'],
                    'Other' : ['Other', 'I do not wish to disclose my location']}

dicc_experiencia = {'Sin experiencia':['I have never written code'],
                    'Junior':['< 1 years', '1-3 years'],
                    'Senior': ['3-5 years', '5-10 years', '10-20 years', '20+ years']}

dicc_dept_size = {'small':['0', '1-2', '3-4'],
                  'medium':['5-9', '10-14'],
                  'large':['15-19', '20+']}

dicc_ML = {'None': ['I do not use machine learning methods'],
           '< 2 years': ['< 1 year', '1-2 years'],
           '2-5 years': ['2-3 years', '3-4 years', '4-5 years'],
           '5+ years': ['5-10 years', '10-20 years', '20 or more years']}

dicc_edad = {'18-29':['18-21', '22-24', '25-29'],
             '30-39':['30-34', '35-39'],
             '40-49':['40-44', '45-49'],
             '50-59':['50-54', '55-59'],
             '60+':['60-69', '70+']}

dicc_columnas = {'country' : dicc_continentes,
                 'years_programming' : dicc_experiencia,
                 'size_DA_dept' : dicc_dept_size,
                 'ML' : dicc_ML,
                 'age' : dicc_edad}

nuevo_orden = ['age',
                'gender',
                'country',
                'continent',
                'job_title',
                'work_activities',
                'size_DA_dept',
                'dept size',
                'years_programming',
                'programming_experience',
                'first_language_rec',
                'dev_language',
                'primary_data_tool',
                'IDE',
                'big_data',
                'BI_tools',
                'visualisation',
                'ML',
                'ML_experience',
                'notebooks_KaggleNotebooks',
                'notebooks_ColabNotebooks',
                'notebooks_AzureNotebooks',
                'notebooks_Paperspace/Gradient',
                'notebooks_Binder/JupyterHub',
                'notebooks_CodeOcean',
                'notebooks_IBMWatsonStudio',
                'notebooks_AmazonSagemakerStudioNotebooks',
                'notebooks_AmazonEMRNotebooks',
                'notebooks_GoogleCloudNotebooks(AIPlatform/VertexAI)',
                'notebooks_GoogleCloudDatalab',
                'notebooks_DatabricksCollaborativeNotebooks',
                'notebooks_Zeppelin/ZeplNotebooks',
                'notebooks_DeepnoteNotebooks',
                'notebooks_ObservableNotebooks',
                'notebooks_None',
                'notebooks_Other',
                'sharing_PlotlyDash',
                'sharing_Streamlit',
                'sharing_NBViewer',
                'sharing_GitHub',
                'sharing_Personalblog',
                'sharing_Kaggle',
                'sharing_Colab',
                'sharing_Shiny',
                'sharing_does_not_share',
                'sharing_Other']

nuevo_orden2 = ['age',
                'gender',
                'country',
                'continent',
                'job_title',
                'work_activities',
                'size_DA_dept',
                'dept size',
                'years_programming',
                'programming_experience',
                'primary_data_tool',
                'first_language_rec',
                'dev_language',
                'IDE',
                'big_data',
                'BI_tools',
                'visualisation',
                'notebooks', 
                'sharing',
                'ML',
                'ML_experience',
                'notebooks_KaggleNotebooks',
                'notebooks_ColabNotebooks',
                'notebooks_AzureNotebooks',
                'notebooks_Paperspace/Gradient',
                'notebooks_Binder/JupyterHub',
                'notebooks_CodeOcean',
                'notebooks_IBMWatsonStudio',
                'notebooks_AmazonSagemakerStudioNotebooks',
                'notebooks_AmazonEMRNotebooks',
                'notebooks_GoogleCloudNotebooks(AIPlatform/VertexAI)',
                'notebooks_GoogleCloudDatalab',
                'notebooks_DatabricksCollaborativeNotebooks',
                'notebooks_Zeppelin/ZeplNotebooks',
                'notebooks_DeepnoteNotebooks',
                'notebooks_ObservableNotebooks',
                'notebooks_None',
                'notebooks_Other',
                'sharing_PlotlyDash',
                'sharing_Streamlit',
                'sharing_NBViewer',
                'sharing_GitHub',
                'sharing_Personalblog',
                'sharing_Kaggle',
                'sharing_Colab',
                'sharing_Shiny',
                'sharing_does_not_share',
                'sharing_Other']

col_drop = ['notebooks_KaggleNotebooks', 
                          'notebooks_ColabNotebooks', 
                          'notebooks_AzureNotebooks', 
                          'notebooks_Paperspace/Gradient', 
                          'notebooks_Binder/JupyterHub', 
                          'notebooks_CodeOcean', 
                          'notebooks_IBMWatsonStudio', 
                          'notebooks_AmazonSagemakerStudioNotebooks', 
                          'notebooks_AmazonEMRNotebooks', 
                          'notebooks_GoogleCloudNotebooks(AIPlatform/VertexAI)', 
                          'notebooks_GoogleCloudDatalab', 
                          'notebooks_DatabricksCollaborativeNotebooks', 
                          'notebooks_Zeppelin/ZeplNotebooks', 
                          'notebooks_DeepnoteNotebooks', 
                          'notebooks_ObservableNotebooks', 
                          'notebooks_None', 
                          'notebooks_Other', 
                          'sharing_PlotlyDash', 
                          'sharing_Streamlit', 
                          'sharing_NBViewer', 
                          'sharing_GitHub', 
                          'sharing_Personalblog', 
                          'sharing_Kaggle', 
                          'sharing_Colab', 
                          'sharing_Shiny', 
                          'sharing_does_not_share', 
                          'sharing_Other']

lista_columnas_dividir = ['dev_language', 
                          'IDE', 
                          'visualisation', 
                          'work_activities', 
                          'big_data', 
                          'BI_tools']

diccionario_nombres_activities = {'work_activities_Analyze and understand data to influence product or business decisions':'activities_analyze_data',
                                'work_activities_Experimentation and iteration to improve existing ML models':'activities_improve_ML',
                                'work_activities_Build prototypes to explore applying machine learning to new areas':'activities_ML_prototypes',
                                'work_activities_None of these activities are an important part of my role at work':'activities_None',
                                'work_activities_Build and/or run the data infrastructure that my business uses for storing; analyzing; and operationalizing data':'activities_data_infrastructure',
                                'work_activities_Build and/or run a machine learning service that operationally improves my product or workflows':'activities_run_ML',
                                'work_activities_Other':'activities_Other',
                                'work_activities_Do research that advances the state of the art of machine learning':'activities_ML_research'}

lista_columnas_reemplazar = ['dev_language_', 
                             'IDE_', 
                             'visualisation_', 
                             'activities_', 
                             'big_data_', 
                             'BI_tools_']

diccionario_respuestas = {'age': ['50-59', '18-29', '40-49', '30-39', '60+'],
                        'gender': ['Man',
                                    'Woman',
                                    'Nonbinary',
                                    'Prefer not to say',
                                    'Prefer to self-describe'],
                        'country': ['Italy',
                                    'Chile',
                                    'Belarus',
                                    'Colombia',
                                    'Nigeria',
                                    'Germany',
                                    'Ghana',
                                    'United Kingdom of Great Britain and Northern Ireland',
                                    'Pakistan',
                                    'Indonesia',
                                    'Viet Nam',
                                    'Thailand',
                                    'Kazakhstan',
                                    'Switzerland',
                                    'Sweden',
                                    'Tunisia',
                                    'China',
                                    'Turkey',
                                    'Ukraine',
                                    'Other',
                                    'Israel',
                                    'I do not wish to disclose my location',
                                    'Norway',
                                    'Czech Republic',
                                    'Brazil',
                                    'Argentina',
                                    'Kenya',
                                    'Greece',
                                    'Belgium',
                                    'United States of America',
                                    'Ireland',
                                    'Nepal',
                                    'France',
                                    'Egypt',
                                    'Peru',
                                    'Russia',
                                    'Netherlands',
                                    'Malaysia',
                                    'Iran',
                                    'Denmark',
                                    'Philippines',
                                    'India',
                                    'Canada',
                                    'United Arab Emirates',
                                    'Portugal',
                                    'Spain',
                                    'South Korea',
                                    'Ethiopia',
                                    'Morocco',
                                    'Saudi Arabia',
                                    'Romania',
                                    'Singapore',
                                    'Australia',
                                    'Uganda',
                                    'Japan',
                                    'Hong Kong (S.A.R.)',
                                    'Ecuador',
                                    'Bangladesh',
                                    'Taiwan',
                                    'South Africa',
                                    'Islamic Republic of...',
                                    'Austria',
                                    'Sri Lanka',
                                    'Poland',
                                    'Iraq',
                                    'Algeria',
                                    'Mexico'],
                        'continent': ['Asia & Oceania', 'America', 'Africa', 'Europe', 'Other'],
                        'job_title': ['Other',
                                    'Program/Project Manager',
                                    'Software Engineer',
                                    'Research Scientist',
                                    'Currently not employed',
                                    'Student',
                                    'Data Scientist',
                                    'Data Analyst',
                                    'Machine Learning Engineer',
                                    'Business Analyst',
                                    'Data Engineer',
                                    'Product Manager',
                                    'Statistician',
                                    'Developer Relations/Advocacy',
                                    'DBA/Database Engineer'],
                        'work_activities': ['Do research that advances the state of the art of machine learning',
                                    'None of these activities are an important part of my role at work',
                                    'Other',
                                    'Experimentation and iteration to improve existing ML models',
                                    'Build prototypes to explore applying machine learning to new areas',
                                    'Analyze and understand data to influence product or business decisions',
                                    'Build and/or run a machine learning service that operationally improves my product or workflows',
                                    'Build and/or run the data infrastructure that my business uses for storing; analyzing; and operationalizing data'],
                        'size_DA_dept': ['3-4', '1-2', '0', '5-9', '10-14', '20+', '15-19'],
                        'dept size': ['small', 'medium', 'large'],
                        'years_programming': ['5-10 years',
                                    '20+ years',
                                    '1-3 years',
                                    '< 1 years',
                                    '3-5 years',
                                    '10-20 years',
                                    'I have never written code'],
                        'programming_experience': ['Senior', 'Junior', 'Sin experiencia'],
                        'primary_data_tool': ['Local development environments (RStudio, JupyterLab, etc.)',
                                    'Advanced statistical software (SPSS, SAS, etc.)',
                                    'Basic statistical software (Microsoft Excel, Google Sheets, etc.)',
                                    'Business intelligence software (Salesforce, Tableau, Spotfire, etc.)',
                                    'Cloud-based data software & APIs (AWS, GCP, Azure, etc.)',
                                    'Other'],
                        'first_language_rec': ['Python',
                                    'SQL',
                                    'R',
                                    'MATLAB',
                                    'C',
                                    'Julia',
                                    'Other',
                                    'C++',
                                    'Javascript',
                                    'Java',
                                    'None',
                                    'Bash',
                                    'Swift'],
                        'dev_language': ['Javascript',
                                    'SQL',
                                    'Swift',
                                    'Other',
                                    'R',
                                    'MATLAB',
                                    'C',
                                    'C++',
                                    'Bash',
                                    'Python',
                                    'Java',
                                    'None',
                                    'Julia'],
                        'IDE': ['RStudio',
                                    'Vim / Emacs',
                                    'Spyder',
                                    'Jupyter (JupyterLab; Jupyter Notebooks; etc)',
                                    'MATLAB',
                                    'Jupyter Notebook',
                                    'Visual Studio Code (VSCode)',
                                    'Notepad++',
                                    'Visual Studio',
                                    'PyCharm',
                                    'MATLAB',
                                    'None',
                                    'Visual Studio Code (VSCode)',
                                    'Visual Studio',
                                    'Other',
                                    'PyCharm',
                                    'Sublime Text',
                                    'Jupyter (JupyterLab; Jupyter Notebooks; etc)',
                                    'RStudio'],
                        'big_data': ['Google Cloud Firestore',
                                    'Amazon DynamoDB',
                                    'Oracle Database',
                                    'Snowflake',
                                    'Microsoft Azure SQL Database',
                                    'Amazon RDS',
                                    'MySQL',
                                    'Google Cloud BigTable',
                                    'Microsoft Azure SQL Database',
                                    'MySQL',
                                    'Amazon Redshift',
                                    'PostgreSQL',
                                    'MongoDB',
                                    'Oracle Database',
                                    'MongoDB',
                                    'Google Cloud SQL',
                                    'Google Cloud BigQuery',
                                    'PostgreSQL',
                                    'Microsoft Azure Cosmos DB',
                                    'Amazon Aurora',
                                    'Microsoft SQL Server',
                                    'None',
                                    'SQLite',
                                    'IBM Db2',
                                    'Google Cloud Spanner',
                                    'Amazon Aurora',
                                    'Amazon Redshift',
                                    'Snowflake',
                                    'Google Cloud Firestore',
                                    'Google Cloud BigTable',
                                    'Other',
                                    'SQLite',
                                    'Amazon RDS',
                                    'Google Cloud Spanner',
                                    'Google Cloud SQL',
                                    'Google Cloud BigQuery',
                                    'Microsoft SQL Server',
                                    'Microsoft Azure Cosmos DB',
                                    'IBM Db2',
                                    'Amazon DynamoDB'], 
                        'BI_tools': ['Salesforce',
                                    'Tableau CRM',
                                    'SAP Analytics Cloud',
                                    'Sisense',
                                    'Tableau',
                                    'TIBCO Spotfire',
                                    'Sisense',
                                    'Microsoft Azure Synapse',
                                    'Microsoft Power BI',
                                    'Looker',
                                    'Domo',
                                    'Amazon QuickSight',
                                    'Alteryx',
                                    'Thoughtspot',
                                    'Thoughtspot',
                                    'None',
                                    'Alteryx',
                                    'SAP Analytics Cloud',
                                    'Other',
                                    'Google Data Studio',
                                    'Qlik',
                                    'Microsoft Azure Synapse'],
                        'visualisation': ['Leaflet / Folium',
                                    'Matplotlib',
                                    'Shiny',
                                    'Leaflet / Folium',
                                    'D3 js',
                                    'Bokeh',
                                    'Seaborn',
                                    'Ggplot / ggplot2',
                                    'Plotly / Plotly Express',
                                    'Bokeh',
                                    'Altair',
                                    'Ggplot / ggplot2',
                                    'Matplotlib',
                                    'D3 js',
                                    'None',
                                    'Seaborn',
                                    'Altair',
                                    'Other',
                                    'Geoplotlib',
                                    'Shiny',
                                    'Plotly / Plotly Express',
                                    'Geoplotlib'],
                        'notebooks': ['Zeppelin / Zepl Notebooks',
                                    'Amazon EMR Notebooks',
                                    'Colab Notebooks',
                                    'Amazon Sagemaker Studio Notebooks',
                                    'None',
                                    'IBM Watson Studio',
                                    'Observable Notebooks',
                                    'Google Cloud Datalab',
                                    'Code Ocean',
                                    'Observable Notebooks',
                                    'Other',
                                    'Zeppelin / Zepl Notebooks',
                                    'Amazon Sagemaker Studio Notebooks',
                                    'Google Cloud Notebooks (AI Platform / Vertex AI)',
                                    'Google Cloud Notebooks (AI Platform / Vertex AI)',
                                    'Paperspace / Gradient',
                                    'Deepnote Notebooks',
                                    'Paperspace / Gradient',
                                    'Binder / JupyterHub',
                                    'Deepnote Notebooks',
                                    'Amazon EMR Notebooks',
                                    'Code Ocean',
                                    'IBM Watson Studio',
                                    'Kaggle Notebooks',
                                    'Binder / JupyterHub',
                                    'Azure Notebooks',
                                    'Databricks Collaborative Notebooks',
                                    'Databricks Collaborative Notebooks'],
                        'sharing': ['Kaggle',
                                    'Shiny',
                                    'Personal blog',
                                    'Other',
                                    'GitHub',
                                    'Colab',
                                    'Kaggle',
                                    'Plotly Dash',
                                    'Plotly Dash',
                                    'NBViewer',
                                    'Personal blog',
                                    'Colab',
                                    'I do not share my work publicly',
                                    'GitHub',
                                    'Shiny',
                                    'Streamlit',
                                    'Streamlit',
                                    'NBViewer'],
                        'ML': ['5-10 years',
                                    '< 1 year',
                                    'I do not use machine learning methods',
                                    '10-20 years',
                                    '2-3 years',
                                    '1-2 years',
                                    '4-5 years',
                                    '3-4 years',
                                    '20 or more years'],
                        'ML_experience': ['5+ years', '< 2 years', 'None', '2-5 years']}