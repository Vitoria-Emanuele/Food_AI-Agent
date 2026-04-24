import kagglehub
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, initcap, concat_ws

def iniciar_sessao_spark():
    print("Iniciando a sessão do PySpark.")
    spark = SparkSession.builder \
        .appName("iFood_Data_Prep") \
        .getOrCreate()
    return spark


def carregar_dados_brutos():
    print("Baixando o dataset do Kaggle.")
    path = kagglehub.dataset_download("benjnb/global-street-food-dataset")
    print(f"Caminho para os arquivos do dataset: {path}")
    
    csv_path = f"{path}/global_street_food.csv"
    return csv_path

def preprocessar_dados(df):
    """Aplica as regras de limpeza de dados """
    print("Iniciando o pré-processamento:")
    
    # remove linhas duplicadas
    df_limpo = df.dropDuplicates(["Dish Name"])
       
    # tratar valores nulos preenchendo com "desconhecido" (para strings)
    df_limpo = df_limpo.fillna("desconhecido")
    
    # formata a coluna "Dish Name" como titulo (Initcap)
    df_limpo = df_limpo.withColumn("Dish Name", initcap(col("Dish Name")))
    
    # unir a descrição, metodo de cozimento e ingredientes com um ponto final
    df_limpo = df_limpo.withColumn(
        "text_for_embedding", 
        concat_ws(" ", col("Description"), col("Cooking Method"), col("Ingredients"))
    )
    
    return df_limpo


if __name__ == "__main__":
    # inicia o spark
    spark = iniciar_sessao_spark()
    
    # caminho do CSV
    caminho_csv = carregar_dados_brutos()
    
    # leitura do CSV 
    print("Carregando dados no DataFrame...")
    df_bruto = spark.read.csv(caminho_csv, header=True, inferSchema=True)

    # funcao de limpeza
    df_processado = preprocessar_dados(df_bruto)
    
    df_processado.select("Dish Name", "text_for_embedding").show(5, truncate=False)

    print("Exportando os dados processados...")
    
    # trazemos os dados do spark para a memoria nativa do python em formato de dicionario
    dados_lista = [row.asDict() for row in df_processado.select("Dish Name", "Typical Price (USD)", "text_for_embedding").collect()]
    
    # salvando o arquivo usando o python puro, ignorando o limitador do PySpark
    import json
    with open("dados_limpos.json", "w", encoding="utf-8") as f:
        for item in dados_lista:
            f.write(json.dumps(item) + "\n")
            
    print("Arquivo 'dados_limpos.json' salvo com sucesso!")