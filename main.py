from hexGame import JogoHex
from qlearning_agent import QlearningAgent, comparar_agentes, plotar_metricas
from utils import listar_arquivos_drive, baixar_arquivo_drive
import os
import threading
import time
from colorama import init, Fore, Style
init(autoreset=True)

CAMINHO_MODELOS = "modelos_baixados" # modelos baixados do drive
os.makedirs(CAMINHO_MODELOS, exist_ok=True)

PASTA_DRIVE_ID = "1_PeNiEZy8jhmNWNFVES3bPXU6g8TNzjn"  # ID da pasta do seu Drive

def menu_principal():
    """Menu principal do jogo com opções de Q-Learning"""
    print("=" * 50)
    print("🎯 BEM-VINDO AO HEX GAME! 🎯")
    print("=" * 50)
    print("1. 🎮 Jogo Tradicional (Humano vs Humano ou vs Minimax)")
    print("2. 🧠 Treinar Agente Q-Learning")
    print("3. 🤖 Jogar contra Agente Q-Learning")
    print("4. ⚔️  Comparar Q-Learning vs Minimax")
    print("5. 📊 Visualizar Métricas de Treinamento")
    print("6. 💾 Gerenciar Modelos Salvos")
    print("0. 🚪 Sair")
    print("=" * 50)

def jogo_tradicional():
    """Executa o jogo tradicional (código original)"""
    print("\n=== JOGO HEX TRADICIONAL ===")
    print(f"O jogador {Fore.RED}⬢{Style.RESET_ALL} (X) deve conectar a esquerda à direita")
    print(f"O jogador {Fore.CYAN}⬢{Style.RESET_ALL} (O) deve conectar o topo à base")
    
    tamanho = 11
    try:
        tamanho_input = int(input("Digite o tamanho do tabuleiro (3-11, padrão: 11): "))
        if 3 <= tamanho_input <= 11:
            tamanho = tamanho_input
        else:
            print("Tamanho inválido! Usando o tamanho padrão de 11x11.")
    except ValueError:
        print("Entrada inválida! Usando o tamanho padrão de 11x11.")
    
    jogo = JogoHex(tamanho)
    jogo.jogar()

def treinar_agente():
    """Interface para treinar um novo agente Q-Learning"""
    print("\n=== TREINAMENTO DO AGENTE Q-LEARNING ===")
    
    # Configurações do tabuleiro
    try:
        tamanho = int(input("Tamanho do tabuleiro (3-11, padrão 7): ") or "7")
        if not (3 <= tamanho <= 11):
            tamanho = 7
    except ValueError:
        tamanho = 7
    
    # Configurações de treinamento
    try:
        num_episodios = int(input("Número de episódios (padrão 5000): ") or "5000")
    except ValueError:
        num_episodios = 5000
    
    # Tipo de oponente
    print("\nTipos de oponente disponíveis:")
    print("1. aleatorio - Jogadas aleatórias (mais rápido)")
    print("2. minimax - Algoritmo Minimax (mais lento, melhor qualidade)")
    
    tipo_oponente = input("Escolha o tipo de oponente (1 ou 2, padrão 1): ")
    if tipo_oponente == "2":
        oponente = "minimax"
    else:
        oponente = "aleatorio"
    
    # Parâmetros do Q-Learning
    print("\n=== PARÂMETROS DO Q-LEARNING ===")
    try:
        alpha = float(input("Taxa de aprendizado α (0.0-1.0, padrão 0.1): ") or "0.1")
        gamma = float(input("Fator de desconto γ (0.0-1.0, padrão 0.9): ") or "0.9")
        epsilon = float(input("Taxa de exploração inicial ε (0.0-1.0, padrão 1.0): ") or "1.0")
    except ValueError:
        alpha, gamma, epsilon = 0.1, 0.9, 1.0
        print("Usando parâmetros padrão: α=0.1, γ=0.9, ε=1.0")
    
    # Cria e treina o agente
    agente = QlearningAgent(
        tamanho_tabuleiro=tamanho,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    # Nome do arquivo
    nome_arquivo = input(f"Nome do arquivo para salvar (padrão qlearning_{tamanho}x{tamanho}_{num_episodios}ep.pkl): ") or f"qlearning_{tamanho}x{tamanho}_{num_episodios}ep.pkl"
    
    print(f"\n🚀 Iniciando treinamento...")
    print(f"📋 Configurações:")
    print(f"   - Tabuleiro: {tamanho}x{tamanho}")
    print(f"   - Episódios: {num_episodios}")
    print(f"   - Oponente: {oponente}")
    print(f"   - Parâmetros: α={alpha}, γ={gamma}, ε={epsilon}")
    print(f"   - Arquivo: {nome_arquivo}")
    
    # Confirma antes de iniciar
    continuar = input("\nDeseja continuar com o treinamento? (s/n): ").lower()
    if continuar != 's':
        print("Treinamento cancelado.")
        return
    
    # Executa o treinamento
    agente.treinar(
        num_episodios=num_episodios,
        oponente=oponente,
        salvar_a_cada=max(100, num_episodios//10),
        nome_arquivo=nome_arquivo
    )
    
    print("\n✅ Treinamento concluído!")
    
    # Opções pós-treinamento
    print("\n=== OPÇÕES PÓS-TREINAMENTO ===")
    print("1. 📊 Visualizar métricas")
    print("2. 🎮 Jogar contra o agente")
    print("3. ⚔️ Comparar com Minimax")
    print("4. 🔙 Voltar ao menu principal")
    
    opcao = input("Escolha uma opção: ")
    
    if opcao == "1":
        plotar_metricas(agente)
    elif opcao == "2":
        agente.jogar_contra_humano()
    elif opcao == "3":
        num_jogos = int(input("Número de jogos para comparação (padrão 50): ") or "50")
        comparar_agentes(agente, num_jogos, tamanho)
    
def jogar_contra_qlearning():
    """Interface para jogar contra um agente Q-Learning treinado - Tabuleiro 7x7"""
    print("\n=== JOGAR CONTRA AGENTE Q-LEARNING (7x7) ===")
    
    # Modelos locais
    print("📁 Buscando modelos locais...")
    modelos_locais = []
    if os.path.exists(CAMINHO_MODELOS):
        modelos_locais = [
            {"name": f, "origem": "local"}
            for f in os.listdir(CAMINHO_MODELOS) if f.endswith(".pkl")
        ]

    # Modelos do Drive
    print("🌐 Buscando modelos no Google Drive...")
    try:
        arquivos_drive = listar_arquivos_drive(PASTA_DRIVE_ID)
        modelos_drive = [
            {"name": f["name"], "id": f["id"], "origem": "drive"}
            for f in arquivos_drive
            if f["name"].endswith(".pkl")
        ]
    except Exception as e:
        print(f"⚠️ Erro ao listar modelos do Drive: {e}")
        modelos_drive = []

    modelos_disponiveis = modelos_locais + modelos_drive
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo disponível!")
        print("💡 Sugestão: Treine um agente primeiro usando a opção 2 do menu principal.")
        return
    
    print("\nModelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        origem = "Drive" if modelo["origem"] == "drive" else "Local"
        print(f"{i}. {modelo['name']} [{origem}]")
    
    try:
        escolha = int(input("Escolha um modelo (número): ")) - 1
        if 0 <= escolha < len(modelos_disponiveis):
            modelo_escolhido = modelos_disponiveis[escolha]
        else:
            print("❌ Escolha inválida!")
            return
    except ValueError:
        print("❌ Entrada inválida! Por favor, digite apenas o número.")
        return
    
    # Carrega o agente
    try:
        agente = QlearningAgent()
        if modelo_escolhido["origem"] == "drive":
            caminho_local = baixar_arquivo_drive(modelo_escolhido["id"], modelo_escolhido["name"])
        else:
            caminho_local = os.path.join(CAMINHO_MODELOS, modelo_escolhido["name"])
        
        agente.carregar_modelo(caminho_local)
        
        # FORÇA O TABULEIRO A SER 7x7 INDEPENDENTE DO MODELO
        agente.tamanho_tabuleiro = 7
        
        print(f"\n✅ Modelo carregado: {modelo_escolhido['name']}")
        print(f"🎯 Tabuleiro configurado para: 7x7")
        print(f"📊 Estatísticas do modelo:")
        total_jogos = agente.vitorias + agente.derrotas
        if total_jogos > 0:
            print(f"   - Vitórias: {agente.vitorias} ({agente.vitorias/total_jogos*100:.1f}%)")
            print(f"   - Derrotas: {agente.derrotas} ({agente.derrotas/total_jogos*100:.1f}%)")
            print(f"   - Tamanho da Q-table: {len(agente.q_table)} entradas")
        
        # Inicia o jogo
        agente.jogar_contra_humano()
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")

def comparar_agentes_interface():
    """Interface para comparar Q-Learning vs Minimax"""
    print("\n=== COMPARAÇÃO Q-LEARNING VS MINIMAX ===")
    
    # Modelos locais
    print("📁 Buscando modelos locais...")
    modelos_locais = []
    if os.path.exists(CAMINHO_MODELOS):
        modelos_locais = [
            {"name": f, "origem": "local"}
            for f in os.listdir(CAMINHO_MODELOS) if f.endswith(".pkl")
        ]

    # Modelos do Drive
    print("🌐 Buscando modelos no Google Drive...")
    try:
        arquivos_drive = listar_arquivos_drive(PASTA_DRIVE_ID)
        modelos_drive = [
            {"name": f["name"], "id": f["id"], "origem": "drive"}
            for f in arquivos_drive
            if f["name"].endswith(".pkl")
        ]
    except Exception as e:
        print(f"⚠️ Erro ao listar modelos do Drive: {e}")
        modelos_drive = []

    modelos_disponiveis = modelos_locais + modelos_drive
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo disponível!")
        return
    
    print("\nModelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        origem = "Drive" if modelo["origem"] == "drive" else "Local"
        print(f"{i}. {modelo['name']} [{origem}]")
    
    try:
        escolha = int(input("Escolha um modelo (número): ")) - 1
        if 0 <= escolha < len(modelos_disponiveis):
            modelo_escolhido = modelos_disponiveis[escolha]
        else:
            print("❌ Escolha inválida!")
            return
    except ValueError:
            print("⚠️ Entrada inválida! Por favor, digite apenas o número da lista.")
            return
    
    try:
        num_jogos = int(input("Número de jogos para comparação (padrão 100): ") or "100")
    except ValueError:
        num_jogos = 100
    
    # Carrega o agente e executa a comparação
    try:
        agente = QlearningAgent()
        if modelo_escolhido["origem"] == "drive":
            caminho_local = baixar_arquivo_drive(modelo_escolhido["id"], modelo_escolhido["name"])
        else:
            caminho_local = os.path.join(CAMINHO_MODELOS, modelo_escolhido["name"])
        agente.carregar_modelo(caminho_local)
        
        print(f"\n🔄 Iniciando comparação com {num_jogos} jogos...")
        print("⏱️ Isso pode demorar alguns minutos...")
        
        vitorias_q, vitorias_minimax, empates = comparar_agentes(
            agente, 
            num_jogos=num_jogos, 
            tamanho_tabuleiro=agente.tamanho_tabuleiro
        )
        
        # Análise detalhada dos resultados
        print(f"\n📊 ANÁLISE DETALHADA:")
        print(f"🎯 Performance do Q-Learning: {vitorias_q/num_jogos*100:.1f}%")
        print(f"🤖 Performance do Minimax: {vitorias_minimax/num_jogos*100:.1f}%")
        print(f"🤝 Empates: {empates} ({empates/num_jogos*100:.1f}%)")
        
        if vitorias_q > vitorias_minimax:
            print("🏆 Q-Learning se saiu melhor!")
        elif vitorias_minimax > vitorias_q:
            print("🏆 Minimax se saiu melhor!")
        else:
            print("🤝 Empate técnico!")
            
    except Exception as e:
        print(f"❌ Erro durante a comparação: {e}")

def visualizar_metricas():
    """Interface para visualizar métricas de treinamento"""
    print("\n=== VISUALIZAÇÃO DE MÉTRICAS ===")
    
    # Modelos locais
    print("📁 Buscando modelos locais...")
    modelos_locais = []
    if os.path.exists(CAMINHO_MODELOS):
        modelos_locais = [
            {"name": f, "origem": "local"}
            for f in os.listdir(CAMINHO_MODELOS) if f.endswith(".pkl")
        ]

    # Modelos do Drive
    print("🌐 Buscando modelos no Google Drive...")
    try:
        arquivos_drive = listar_arquivos_drive(PASTA_DRIVE_ID)
        modelos_drive = [
            {"name": f["name"], "id": f["id"], "origem": "drive"}
            for f in arquivos_drive
            if f["name"].endswith(".pkl")
        ]
    except Exception as e:
        print(f"⚠️ Erro ao listar modelos do Drive: {e}")
        modelos_drive = []

    modelos_disponiveis = modelos_locais + modelos_drive

    if not modelos_disponiveis:
        print("❌ Nenhum modelo Q-Learning encontrado!")
        return

    # Lista todos os modelos
    print("\nModelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        origem = "Drive" if modelo["origem"] == "drive" else "Local"
        print(f"{i}. {modelo['name']} [{origem}]")

    try:
        escolha = int(input("Escolha um modelo (número): ")) - 1
        if 0 <= escolha < len(modelos_disponiveis):
            modelo = modelos_disponiveis[escolha]
            if modelo["origem"] == "drive":
                caminho_local = baixar_arquivo_drive(modelo["id"], modelo["name"])
            else:
                caminho_local = os.path.join(CAMINHO_MODELOS, modelo["name"])
        else:
            print("❌ Escolha inválida!")
            return
    except ValueError:
        print("❌ Entrada inválida.")
        return

    try:
        agente = QlearningAgent()
        agente.carregar_modelo(caminho_local)
        plotar_metricas(agente)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")


def gerenciar_modelos():
    """Interface para gerenciar modelos salvos locais e do Google Drive"""
    print("\n=== GERENCIAMENTO DE MODELOS ===")

    # Modelos locais
    modelos_locais = []
    if os.path.exists(CAMINHO_MODELOS):
        modelos_locais = [
            {"name": f, "origem": "local"}
            for f in os.listdir(CAMINHO_MODELOS) if f.endswith(".pkl")
        ]

    # Modelos no Google Drive
    try:
        arquivos_drive = listar_arquivos_drive(PASTA_DRIVE_ID)
        modelos_drive = [
            {"name": f["name"], "id": f["id"], "origem": "drive"}
            for f in arquivos_drive if f["name"].endswith(".pkl")
        ]
    except Exception as e:
        print(f"⚠️ Erro ao listar modelos do Drive: {e}")
        modelos_drive = []

    modelos_disponiveis = modelos_locais + modelos_drive

    if not modelos_disponiveis:
        print("❌ Nenhum modelo encontrado!")
        return

    print("📁 Modelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        origem = "Drive" if modelo["origem"] == "drive" else "Local"
        print(f"{i}. {modelo['name']} [{origem}]")

    print("\n=== OPÇÕES ===")
    print("1. 📊 Ver detalhes de um modelo")
    print("2. 🗑️  Deletar um modelo (apenas local)")
    print("3. 📄 Renomear um modelo (apenas local)")
    print("4. 🔙 Voltar ao menu principal")

    opcao = input("Escolha uma opção: ")

    if opcao == "1":
        try:
            escolha = int(input("Escolha um modelo para ver detalhes (número): ")) - 1
            if 0 <= escolha < len(modelos_disponiveis):
                modelo = modelos_disponiveis[escolha]

                if modelo["origem"] == "drive":
                    caminho_local = baixar_arquivo_drive(modelo["id"], modelo["name"])
                else:
                    caminho_local = os.path.join(CAMINHO_MODELOS, modelo["name"])

                agente = QlearningAgent()
                agente.carregar_modelo(caminho_local)

                print(f"\n📊 DETALHES DO MODELO: {modelo['name']}")
                print(f"🎯 Tamanho do tabuleiro: {agente.tamanho_tabuleiro}x{agente.tamanho_tabuleiro}")
                print(f"🧠 Tamanho da Q-table: {len(agente.q_table)} entradas")
                print(f"📈 Parâmetros: α={agente.alpha}, γ={agente.gamma}")

                total_jogos = agente.vitorias + agente.derrotas
                if total_jogos > 0:
                    print(f"🏆 Vitórias: {agente.vitorias} ({agente.vitorias/total_jogos*100:.1f}%)")
                    print(f"💔 Derrotas: {agente.derrotas} ({agente.derrotas/total_jogos*100:.1f}%)")
                    print(f"🎮 Total de episódios: {total_jogos}")
        except Exception as e:
            print(f"❌ Erro: {e}")

    elif opcao == "2":
        try:
            escolha = int(input("Escolha um modelo local para deletar (número): ")) - 1
            if 0 <= escolha < len(modelos_disponiveis):
                modelo = modelos_disponiveis[escolha]
                if modelo["origem"] == "local":
                    confirmar = input(f"❗ Tem certeza que deseja deletar '{modelo['name']}'? (s/n): ").lower()
                    if confirmar == 's':
                        os.remove(os.path.join(CAMINHO_MODELOS, modelo["name"]))
                        print(f"✅ Modelo '{modelo['name']}' deletado com sucesso!")
                    else:
                        print("❌ Operação cancelada.")
                else:
                    print("❌ Não é possível deletar arquivos do Drive.")
        except Exception as e:
            print(f"❌ Erro ao deletar: {e}")

    elif opcao == "3":
        try:
            escolha = int(input("Escolha um modelo local para renomear (número): ")) - 1
            if 0 <= escolha < len(modelos_disponiveis):
                modelo = modelos_disponiveis[escolha]
                if modelo["origem"] == "local":
                    nome_antigo = modelo["name"]
                    nome_novo = input("Digite o novo nome (sem extensão): ")
                    if not nome_novo.endswith('.pkl'):
                        nome_novo += '.pkl'

                    os.rename(
                        os.path.join(CAMINHO_MODELOS, nome_antigo),
                        os.path.join(CAMINHO_MODELOS, nome_novo)
                    )
                    print(f"✅ Modelo renomeado de '{nome_antigo}' para '{nome_novo}'!")
                else:
                    print("❌ Não é possível renomear arquivos do Drive.")
        except Exception as e:
            print(f"❌ Erro ao renomear: {e}")

    elif opcao == "4":
        return

    else:
        print("❌ Opção inválida.")

def main():
    """Função principal com loop do menu"""
    while True:
        try:
            menu_principal()
            opcao = input("Escolha uma opção (0-6): ").strip()
            
            if opcao == "0":
                print("\n👋 Obrigado por jogar! Até logo!")
                break
            elif opcao == "1":
                jogo_tradicional()
            elif opcao == "2":
                treinar_agente()
            elif opcao == "3":
                jogar_contra_qlearning()
            elif opcao == "4":
                comparar_agentes_interface()
            elif opcao == "5":
                visualizar_metricas()
            elif opcao == "6":
                gerenciar_modelos()
            else:
                print("❌ Opção inválida! Escolha um número de 0 a 6.")
            
            # Pausa antes de voltar ao menu
            if opcao != "0":
                input("\n📄 Pressione Enter para voltar ao menu principal...")
                print("\n" + "="*50)
        
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro inesperado: {e}")
            print("🔄 Voltando ao menu principal...")

if __name__ == "__main__":
    print("🎯 Inicializando Hex Game...")
    main()