"""
Módulo: API REST 
Descrição: Expoe o nosso Agente de IA em um endpoint (stateless) 
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vector_db import DocumentVectorSearch
import uvicorn

app = FastAPI(title="AI Agent - Recomendação de Pratos")

buscador = DocumentVectorSearch()

try:
    print("Inicializando motor de IA.")
    buscador.load_documents("dados_limpos.json")
    buscador.process_documents()
    print("Sistema pronto para recomendações.")
except Exception as e:
    print(f"Erro crítico ao carregar dados: {e}")

class PedidoCliente(BaseModel):
    mensagem: str

@app.post("/sugerir-prato")
async def sugerir_prato(pedido: PedidoCliente):
    if not pedido.mensagem.strip():
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")
    
    try:
        resultados_faiss = buscador.query(pedido.mensagem, num_results=3)
        
        return {
            "status": "sucesso",
            "mensagem_agente": "Encontramos estas opções para você:",
            "recomendacoes": resultados_faiss
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)