from hexGame import JogoHex
from qlearning_agent import QlearningAgent, comparar_agentes, plotar_metricas
import os
import threading
import time

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
    print("O jogador X deve conectar a esquerda à direita")
    print("O jogador O deve conectar o topo à base")
    
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
    """Interface para jogar contra um agente Q-Learning treinado"""
    print("\n=== JOGAR CONTRA AGENTE Q-LEARNING ===")
    
    # Lista modelos disponíveis
    modelos_disponiveis = [f for f in os.listdir('.') if f.endswith('.pkl') and 'qlearning' in f.lower()]
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo Q-Learning encontrado!")
        print("💡 Dica: Treine um agente primeiro usando a opção 2 do menu principal.")
        return
    
    print("📁 Modelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        print(f"{i}. {modelo}")
    
    try:
        escolha = int(input("Escolha um modelo (número): ")) - 1
        if 0 <= escolha < len(modelos_disponiveis):
            nome_arquivo = modelos_disponiveis[escolha]
        else:
            print("Escolha inválida!")
            return
    except ValueError:
        nome_arquivo = input("Digite o nome do arquivo: ")
        if not nome_arquivo.endswith('.pkl'):
            nome_arquivo += '.pkl'
    
    # Carrega o agente
    try:
        # Cria um agente temporário para carregar o modelo
        agente = QlearningAgent()
        agente.carregar_modelo(nome_arquivo)
        
        print(f"\n✅ Modelo carregado: {nome_arquivo}")
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
    
    # Lista modelos disponíveis
    modelos_disponiveis = [f for f in os.listdir('.') if f.endswith('.pkl') and 'qlearning' in f.lower()]
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo Q-Learning encontrado!")
        print("💡 Dica: Treine um agente primeiro usando a opção 2 do menu principal.")
        return
    
    print("📁 Modelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        print(f"{i}. {modelo}")
    
    try:
        escolha = int(input("Escolha um modelo (número): ")) - 1
        if 0 <= escolha < len(modelos_disponiveis):
            nome_arquivo = modelos_disponiveis[escolha]
        else:
            print("Escolha inválida!")
            return
    except ValueError:
        nome_arquivo = input("Digite o nome do arquivo: ")
        if not nome_arquivo.endswith('.pkl'):
            nome_arquivo += '.pkl'
    
    try:
        num_jogos = int(input("Número de jogos para comparação (padrão 100): ") or "100")
    except ValueError:
        num_jogos = 100
    
    # Carrega o agente e executa a comparação
    try:
        agente = QlearningAgent()
        agente.carregar_modelo(nome_arquivo)
        
        print(f"\n🔄 Iniciando comparação com {num_jogos} jogos...")
        print("⏱️ Isso pode demorar alguns minutos...")
        
        vitorias_q, vitorias_minimax = comparar_agentes(
            agente, 
            num_jogos=num_jogos, 
            tamanho_tabuleiro=agente.tamanho_tabuleiro
        )
        
        # Análise detalhada dos resultados
        print(f"\n📊 ANÁLISE DETALHADA:")
        print(f"🎯 Performance do Q-Learning: {vitorias_q/num_jogos*100:.1f}%")
        print(f"🤖 Performance do Minimax: {vitorias_minimax/num_jogos*100:.1f}%")
        
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
    
    # Lista modelos disponíveis
    modelos_disponiveis = [f for f in os.listdir('.') if f.endswith('.pkl') and 'qlearning' in f.lower()]
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo Q-Learning encontrado!")
        return
    
    print("📁 Modelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        print(f"{i}. {modelo}")
    
    try:
        escolha = int(input("Escolha um modelo (número): ")) - 1
        if 0 <= escolha < len(modelos_disponiveis):
            nome_arquivo = modelos_disponiveis[escolha]
        else:
            print("Escolha inválida!")
            return
    except ValueError:
        nome_arquivo = input("Digite o nome do arquivo: ")
        if not nome_arquivo.endswith('.pkl'):
            nome_arquivo += '.pkl'
    
    try:
        agente = QlearningAgent()
        agente.carregar_modelo(nome_arquivo)
        plotar_metricas(agente)
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")

def gerenciar_modelos():
    """Interface para gerenciar modelos salvos"""
    print("\n=== GERENCIAMENTO DE MODELOS ===")
    
    # Lista todos os arquivos .pkl
    modelos_disponiveis = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo encontrado!")
        return
    
    print("📁 Modelos disponíveis:")
    for i, modelo in enumerate(modelos_disponiveis, 1):
        tamanho_arquivo = os.path.getsize(modelo) / (1024*1024)  # MB
        print(f"{i}. {modelo} ({tamanho_arquivo:.1f} MB)")
    
    print("\n=== OPÇÕES ===")
    print("1. 📊 Ver detalhes de um modelo")
    print("2. 🗑️ Deletar um modelo")
    print("3. 📄 Renomear um modelo")
    print("4. 🔙 Voltar ao menu principal")
    
    opcao = input("Escolha uma opção: ")
    
    if opcao == "1":
        try:
            escolha = int(input("Escolha um modelo para ver detalhes (número): ")) - 1
            if 0 <= escolha < len(modelos_disponiveis):
                nome_arquivo = modelos_disponiveis[escolha]
                agente = QlearningAgent()
                agente.carregar_modelo(nome_arquivo)
                
                print(f"\n📊 DETALHES DO MODELO: {nome_arquivo}")
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
            escolha = int(input("Escolha um modelo para deletar (número): ")) - 1
            if 0 <= escolha < len(modelos_disponiveis):
                nome_arquivo = modelos_disponiveis[escolha]
                confirmar = input(f"❗ Tem certeza que deseja deletar '{nome_arquivo}'? (s/n): ").lower()
                if confirmar == 's':
                    os.remove(nome_arquivo)
                    print(f"✅ Modelo '{nome_arquivo}' deletado com sucesso!")
                else:
                    print("❌ Operação cancelada.")
        except Exception as e:
            print(f"❌ Erro ao deletar: {e}")
    
    elif opcao == "3":
        try:
            escolha = int(input("Escolha um modelo para renomear (número): ")) - 1
            if 0 <= escolha < len(modelos_disponiveis):
                nome_antigo = modelos_disponiveis[escolha]
                nome_novo = input("Digite o novo nome (sem extensão): ")
                if not nome_novo.endswith('.pkl'):
                    nome_novo += '.pkl'
                
                os.rename(nome_antigo, nome_novo)
                print(f"✅ Modelo renomeado de '{nome_antigo}' para '{nome_novo}'!")
        except Exception as e:
            print(f"❌ Erro ao renomear: {e}")

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