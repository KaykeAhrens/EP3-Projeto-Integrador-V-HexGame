import os
import requests

def listar_arquivos_drive(pasta_id):
    """Lista arquivos em uma pasta pública do Google Drive (sem autenticação)."""
    url = f"https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{pasta_id}' in parents and trashed = false",
        "fields": "files(id, name)",
        "key": "AIzaSyAnBzQvGX3PkrhkV3z_UUeZbbGyukFZR4Q"  # Você precisa gerar sua chave pública de API do Google aqui
    }
    resposta = requests.get(url, params=params)

    if resposta.status_code != 200:
        raise Exception(f"Erro ao listar arquivos: {resposta.text}")

    return resposta.json()["files"]

def baixar_arquivo_drive(arquivo_id, nome_destino, destino_local="modelos_baixados"):
    """Baixa um arquivo específico do Google Drive pelo ID."""
    os.makedirs(destino_local, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={arquivo_id}"
    resposta = requests.get(url)

    caminho_completo = os.path.join(destino_local, nome_destino)
    with open(caminho_completo, 'wb') as f:
        f.write(resposta.content)

    return caminho_completo


