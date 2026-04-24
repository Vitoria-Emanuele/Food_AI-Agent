import kagglehub
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, initcap

def iniciar_sessao_spark():
    print("Iniciando a sessão do PySpark...")
    spark = SparkSession.builder \
        .appName("iFood_Data_Prep") \
        .getOrCreate()
    return spark

def carregar_dados_brutos():
    print("Baixando o dataset do Kaggle...")
    path = kagglehub.dataset_download("benjnb/global-street-food-dataset")
    print(f"Caminho para os arquivos do dataset: {path}")
    
    csv_path = f"{path}/global_street_food.csv"
    return csv_path


if __name__ == "__main__":
    # inicia o spark
    spark = iniciar_sessao_spark()
    
    # caminho do CSV
    caminho_csv = carregar_dados_brutos()
    
    # leitura do CSV 
    print("Carregando dados no DataFrame...")
    df_bruto = spark.read.csv(caminho_csv, header=True, inferSchema=True)
    
    df_bruto.show(5)