"""
Módulo: Banco Vetorial e Similaridade.
Descrição: Carrega  o dataset limpo pelo spark, gera embeddings 
com SentenceTransformers e realiza busca vetorial no FAISS.
"""

import json
import glob
import faiss
from sentence_transformers import SentenceTransformer

class DocumentVectorSearch:
    def __init__(self):
        # usando um modelo rapido e leve para codificar em ingles/multilingue
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = [] 

    def load_documents(self, file_path):
        """Carrega o arquivo JSON exportado na etapa de dados."""
        print(f"Lendo dados preparados do arquivo: {file_path}...")
        # le o arquivo linha por linha (formato JSON Lines que salvamos)
        with open(file_path, 'r', encoding='utf-8') as f:
            for linha in f:
                self.documents.append(json.loads(linha))
        print(f"{len(self.documents)} pratos carregados com sucesso!")

    def process_documents(self):
        """Cria e armazena embeddings no FAISS."""
        print("Gerando vetores matemáticos (Embeddings)...")
        textos = [doc["text_for_embedding"] for doc in self.documents]

        # converte para numpy array em float32 (exigencia do FAISS)
        embeddings = self.model.encode(textos, convert_to_numpy=True).astype('float32')

        # cria o indice de similaridade (baseado na distancia Euclidiana (L2))
        dimensao = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimensao)
        self.index.add(embeddings)
        print("FAISS indexado! ")

    def query(self, question, num_results=3):
        """transforma o texto do usuario em vetor e busca os similares."""
        vetor_pergunta = self.model.encode([question], convert_to_numpy=True).astype('float32')
        
        distancias, indices = self.index.search(vetor_pergunta, num_results)
        
        print(f"\nResultados RAG para a busca: '{question}'")
        resultados = []
        
        for i in range(num_results):
            posicao_no_dataset = indices[0][i]
            prato = self.documents[posicao_no_dataset]
            
            resultados.append(prato)
            print(f"- {prato['Dish Name']} | USD {prato['Typical Price (USD)']}")
            print(f"  Resumo: {prato['text_for_embedding'][:80]}...")
        
        return resultados
