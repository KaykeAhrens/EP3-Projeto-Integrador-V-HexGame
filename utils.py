import os
import requests
import gdown

def listar_arquivos_drive(pasta_id):
    """Lista arquivos em uma pasta pública do Google Drive (sem autenticação)."""
    url = f"https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{pasta_id}' in parents and trashed = false",
        "fields": "files(id, name)",
        "key": "AIzaSyAnBzQvGX3PkrhkV3z_UUeZbbGyukFZR4Q"  
    }
    resposta = requests.get(url, params=params)

    if resposta.status_code != 200:
        raise Exception(f"Erro ao listar arquivos: {resposta.text}")

    return resposta.json()["files"]

def baixar_arquivo_drive(arquivo_id, nome_destino, destino_local="modelos_baixados"):
    """
    Baixa um arquivo do Google Drive pelo ID usando gdown,
    que já lida com confirm tokens para arquivos grandes.
    """
    os.makedirs(destino_local, exist_ok=True)
    url = f"https://drive.google.com/uc?id={arquivo_id}"
    caminho_completo = os.path.join(destino_local, nome_destino)
    # gdown cuidará de todo o processo de confirmação
    gdown.download(url, caminho_completo, quiet=False)
    return caminho_completo



