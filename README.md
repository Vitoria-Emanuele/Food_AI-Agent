--------------------------------------------------------------------------------
# AI Agent - Sistema de Recomendação Semantica de Pratos

## Sobre o Projeto
Este é um projeto pratico de um Agente autônomo de IA desenhado para atuar no primeiro passo de um funil de recomendação (Busca Semântica/Retrieval). 

O agente recebe requisições em linguagem natural (ex: *"I want a spicy Mexican food with cheese and tortilla"*) e sugere pratos alternativos baseados no perfil de sabor e ingredientes, utilizando processamento de linguagem natural (NLP) e busca vetorial.

Os dados base utilizados para a inteligência vêm do [Global Street Food Dataset](https://www.kaggle.com/datasets/benjnb/global-street-food-dataset), que contém 4.500 pratos de rua de mais de 70 países.

## Tecnologias Utilizadas
* **Engenharia de Dados:** `PySpark` (Limpeza, formatação e concatenação de strings)
* **Inteligência Artificial:** `SentenceTransformers` (Embeddings) e `FAISS` (Banco Vetorial / RAG)
* **Engenharia de Software (Backend):** `FastAPI` e `Uvicorn` (API REST Stateless)

---

## Arquitetura do Sistema 
A arquitetura foi dividida em três módulos independentes, respeitando o fluxo de *Input > Processamento > Ação*:

### 1. Preparação de Dados (`data_prep.py`)
Script em PySpark responsável por baixar os dados brutos via Kagglehub e realizar o pré-processamento (Lazy Evaluation).
* Tratamento de valores nulos e padronização (Initcap).
* Criação de uma coluna rica (`text_for_embedding`) unindo descrição, método de preparo e ingredientes.
* Exportação contornando limitações do Hadoop no Windows, salvando nativamente em `dados_limpos.json`.

### 2. Motor de Inteligência Artificial (`vector_db.py`)
Módulo que carrega os dados processados e gera a matemática do sistema.
* Utiliza o modelo `all-MiniLM-L6-v2` para transformar as descrições dos pratos em vetores matemáticos (Embeddings) em *float32*.
* Armazena os vetores no índice `IndexFlatL2` do FAISS, permitindo buscas semânticas ultrarrápidas baseadas na distância Euclidiana.

### 3. API REST (`main.py`)
Expõe o cérebro do FAISS através de um endpoint `POST` utilizando FastAPI. Isso permite que qualquer front-end (como um aplicativo mobile) envie um JSON com o pedido do cliente e receba as top 3 sugestões estruturadas em tempo real.

---

## Decisão de Arquitetura & Trade-offs
**Otimização do Banco Vetorial (Deduplicação de Pratos):**
Durante o desenvolvimento, observou-se que o dataset mantinha múltiplas cópias do mesmo prato (ex: *Quesadilla*) devido a variações de preço e região de coleta. 
* **O Problema :** Como o FAISS gera vetores baseados nos ingredientes e modo de preparo, ele estava armazenando vetores idênticos, desperdiçando memória RAM e enviesando as recomendações (sugerindo o mesmo prato três vezes).
* **A Solução (Trade-off) :** Foi aplicada uma regra rígida no PySpark (`dropDuplicates(["Dish Name"])`) sacrificando o histórico de preços e localização no banco vetorial. 
* **Justificativa :** Esta decisão otimiza a IA para o foco exclusivo em **Similaridade Semântica (Retrieval)**, garantindo diversidade nas sugestões.

---