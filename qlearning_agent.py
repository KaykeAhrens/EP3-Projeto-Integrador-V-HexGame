import numpy as np
import pickle
import os
import threading
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from hexGame import JogoHex
from utils import listar_arquivos_drive, baixar_arquivo_drive
from colorama import init, Fore, Style
init(autoreset=True)

CAMINHO_MODELOS = "modelos_baixados" # modelos baixados do drive
os.makedirs(CAMINHO_MODELOS, exist_ok=True)

PASTA_DRIVE_ID = "1_PeNiEZy8jhmNWNFVES3bPXU6g8TNzjn"  # ID da pasta do seu Drive

class QlearningAgent:
    def __init__(self, tamanho_tabuleiro=11, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa o agente Q-Learning
        
        Args:
            tamanho_tabuleiro: Tamanho do tabuleiro Hex
            alpha: Taxa de aprendizado
            gamma: Fator de desconto
            epsilon: Taxa de exploração inicial
            epsilon_decay: Taxa de decaimento do epsilon
            epsilon_min: Valor mínimo do epsilon
        """
        self.tamanho_tabuleiro = tamanho_tabuleiro
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(float)  # Q(estado, ação) = valor
        self.jogador = None  # Será definido durante o treinamento
        
        # Métricas de desempenho
        self.vitorias = 0
        self.derrotas = 0
        self.empates = 0
        self.recompensas_episodio = []
        self.historico_epsilon = []
        
    def codificar_estado(self, jogo):
        """
        Converte o estado do jogo em uma representação hashable
        
        Args:
            jogo: Instância do JogoHex
            
        Returns:
            Tupla representando o estado do jogo
        """
        # Converte o tabuleiro para uma tupla hashable
        tabuleiro_tuple = tuple(tuple(linha) for linha in jogo.tabuleiro)
        jogador_atual = jogo.jogador_atual
        return (tabuleiro_tuple, jogador_atual)
    
    def obter_acoes_possiveis(self, jogo):
        """
        Retorna todas as ações possíveis (posições vazias) no estado atual
        
        Args:
            jogo: Instância do JogoHex
            
        Returns:
            Lista de tuplas (linha, coluna) representando ações possíveis
        """
        acoes = []
        for linha in range(jogo.tamanho):
            for coluna in range(jogo.tamanho):
                if jogo.tabuleiro[linha, coluna] == '⬡':
                    acoes.append((linha, coluna))
        return acoes
    
    def selecionar_acao(self, jogo, treinamento=True):
        """
        Seleciona uma ação usando estratégia epsilon-greedy
        
        Args:
            jogo: Instância do JogoHex
            treinamento: Se True, usa epsilon-greedy; se False, sempre greedy
            
        Returns:
            Tupla (linha, coluna) representando a ação selecionada
        """
        acoes_possiveis = self.obter_acoes_possiveis(jogo)
        
        if not acoes_possiveis:
            return None
        
        # Durante o treinamento, usa epsilon-greedy
        if treinamento and np.random.random() < self.epsilon:
            # Exploração: ação aleatória
            return acoes_possiveis[np.random.randint(len(acoes_possiveis))]
        
        # Exploração: escolhe a melhor ação conhecida
        estado = self.codificar_estado(jogo)
        melhor_acao = None
        melhor_valor = float('-inf')
        
        for acao in acoes_possiveis:
            valor_q = self.q_table[(estado, acao)]
            if valor_q > melhor_valor:
                melhor_valor = valor_q
                melhor_acao = acao
        
        # Se todas as ações têm o mesmo valor (início do treinamento),
        # escolhe uma aleatória
        if melhor_acao is None:
            melhor_acao = acoes_possiveis[np.random.randint(len(acoes_possiveis))]
            
        return melhor_acao
    
    def calcular_recompensa(self, jogo, acao, resultado):
        """
        Calcula a recompensa para uma ação específica
        
        Args:
            jogo: Instância do JogoHex
            acao: Ação tomada (linha, coluna)
            resultado: 'vitoria', 'derrota', 'empate', ou 'continua'
            
        Returns:
            Valor da recompensa
        """
        if resultado == 'vitoria':
            return 100
        elif resultado == 'derrota':
            return -100
        elif resultado == 'empate':
            return 0
        else:
            # Recompensa baseada na utilidade do estado
            utilidade = jogo.calcular_utilidade(self.jogador)
            return utilidade * 0.1  # Pequena recompensa baseada na posição
    
    def atualizar_q_table(self, estado_anterior, acao, recompensa, novo_estado, jogo):
        """
        Atualiza a tabela Q usando a equação do Q-Learning
        
        Args:
            estado_anterior: Estado antes da ação
            acao: Ação tomada
            recompensa: Recompensa recebida
            novo_estado: Estado após a ação
            jogo: Instância do JogoHex atual
        """
        # Valor Q atual
        q_atual = self.q_table[(estado_anterior, acao)]
        
        # Melhor valor Q do próximo estado
        acoes_futuras = self.obter_acoes_possiveis(jogo)
        if acoes_futuras:
            max_q_futuro = max(self.q_table[(novo_estado, acao_futura)] 
                              for acao_futura in acoes_futuras)
        else:
            max_q_futuro = 0
        
        # Atualização Q-Learning
        novo_q = q_atual + self.alpha * (recompensa + self.gamma * max_q_futuro - q_atual)
        self.q_table[(estado_anterior, acao)] = novo_q
    
    def treinar_episodio(self, oponente='aleatorio'):
        """
        Executa um episódio de treinamento
        
        Args:
            oponente: Tipo de oponente ('aleatorio', 'minimax', 'qlearning')
            
        Returns:
            Resultado do episódio ('vitoria', 'derrota', 'empate')
        """
        jogo = JogoHex(self.tamanho_tabuleiro)
        
        # Define quem é X e quem é O aleatoriamente
        if np.random.random() < 0.5:
            self.jogador = 'X'
            oponente_jogador = 'O'
        else:
            self.jogador = 'O'
            oponente_jogador = 'X'
        
        recompensa_total = 0
        historico_jogadas = []  # Para armazenar (estado, ação) do agente
        
        while True:
            if jogo.jogador_atual == self.jogador:
                # Turno do agente Q-Learning
                estado_atual = self.codificar_estado(jogo)
                acao = self.selecionar_acao(jogo, treinamento=True)
                
                if acao is None:
                    break
                
                historico_jogadas.append((estado_atual, acao))
                linha, coluna = acao
                
                # Executa a jogada
                jogo.tabuleiro[linha, coluna] = self.jogador
                
                # Verifica se ganhou
                if jogo.verificar_vitoria(self.jogador):
                    resultado = 'vitoria'
                    recompensa = self.calcular_recompensa(jogo, acao, resultado)
                    recompensa_total += recompensa
                    
                    # Atualiza Q-table para todas as jogadas do episódio
                    for i, (estado, acao_historico) in enumerate(historico_jogadas):
                        if i == len(historico_jogadas) - 1:
                            # Última jogada
                            self.q_table[(estado, acao_historico)] += self.alpha * recompensa
                        else:
                            # Propaga a recompensa para trás
                            self.q_table[(estado, acao_historico)] += self.alpha * (recompensa * (self.gamma ** (len(historico_jogadas) - i - 1)))
                    
                    self.vitorias += 1
                    return resultado
                
                jogo.jogador_atual = oponente_jogador
                
            else:
                # Turno do oponente
                if oponente == 'aleatorio':
                    acoes_oponente = self.obter_acoes_possiveis(jogo)
                    if not acoes_oponente:
                        break
                    linha, coluna = acoes_oponente[np.random.randint(len(acoes_oponente))]
                elif oponente == 'minimax':
                    jogo.jogador_atual = oponente_jogador
                    linha, coluna = jogo.melhor_jogada_ia()
                
                # Executa a jogada do oponente
                jogo.tabuleiro[linha, coluna] = oponente_jogador
                
                # Verifica se o oponente ganhou
                if jogo.verificar_vitoria(oponente_jogador):
                    resultado = 'derrota'
                    recompensa = self.calcular_recompensa(jogo, None, resultado)
                    
                    # Penaliza todas as jogadas do episódio
                    for i, (estado, acao_historico) in enumerate(historico_jogadas):
                        self.q_table[(estado, acao_historico)] += self.alpha * (recompensa * (self.gamma ** (len(historico_jogadas) - i - 1)))
                    
                    self.derrotas += 1
                    return resultado
                
                jogo.jogador_atual = self.jogador
            
            # Verifica empate (tabuleiro cheio)
            if len(self.obter_acoes_possiveis(jogo)) == 0:
                self.empates += 1
                return 'empate'
        
        return 'empate'
    
    def treinar(self, num_episodios=10000, oponente='aleatorio', salvar_a_cada=1000, nome_arquivo='qlearning_agent.pkl'):
        """
        Treina o agente por um número específico de episódios
        
        Args:
            num_episodios: Número de episódios de treinamento
            oponente: Tipo de oponente para treinar contra
            salvar_a_cada: Salva o modelo a cada N episódios
            nome_arquivo: Nome do arquivo para salvar o modelo
        """
        print(f"Iniciando treinamento com {num_episodios} episódios contra oponente {oponente}")
        print(f"Parâmetros: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        
        inicio_tempo = time.time()
        
        for episodio in range(num_episodios):
            resultado = self.treinar_episodio(oponente)
            
            # Decai epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.historico_epsilon.append(self.epsilon)
            
            # Log de progresso
            if (episodio + 1) % 100 == 0:
                taxa_vitoria = self.vitorias / (episodio + 1) * 100
                print(f"Episódio {episodio + 1}/{num_episodios} | "
                      f"Vitórias: {self.vitorias} ({taxa_vitoria:.1f}%) | "
                      f"Derrotas: {self.derrotas} | "
                      f"Empates: {self.empates} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Q-table size: {len(self.q_table)}")
            
            # Salva periodicamente
            if (episodio + 1) % salvar_a_cada == 0:
                self.salvar_modelo(f"{nome_arquivo.split('.')[0]}_episodio_{episodio + 1}.pkl")
                print(f"Modelo salvo após {episodio + 1} episódios")
        
        tempo_total = time.time() - inicio_tempo
        print(f"\nTreinamento concluído em {tempo_total:.2f} segundos")
        print(f"Taxa final de vitórias: {self.vitorias / num_episodios * 100:.2f}%")
        
        # Salva modelo final
        self.salvar_modelo(nome_arquivo)
    
    def salvar_modelo(self, nome_arquivo):
        """Salva o modelo treinado de forma assíncrona usando pickle"""
        def salvar():
            try:
                dados = {
                    'q_table': dict(self.q_table),
                    'parametros': {
                        'tamanho_tabuleiro': self.tamanho_tabuleiro,
                        'alpha': self.alpha,
                        'gamma': self.gamma,
                        'epsilon': self.epsilon,
                        'epsilon_decay': self.epsilon_decay,
                        'epsilon_min': self.epsilon_min
                    },
                    'metricas': {
                        'vitorias': self.vitorias,
                        'derrotas': self.derrotas,
                        'empates': self.empates,
                        'historico_epsilon': self.historico_epsilon
                    }
                }

                caminho = r"G:\Meu Drive\HexGameQTable"
                os.makedirs(caminho, exist_ok=True)

                destino = os.path.join(caminho, nome_arquivo)
                with open(destino, 'wb') as f:
                    pickle.dump(dados, f)
                print(f"\n[✓] Modelo salvo assíncronamente em: {destino}\n")
            except Exception as e:
                print(f"\n[Erro ao salvar modelo assíncrono]: {e}\n")

        # Inicia o salvamento em uma nova thread
        threading.Thread(target=salvar).start()
    
    def carregar_modelo(self, nome_arquivo):
        """Carrega um modelo salvo"""
        try:
            with open(nome_arquivo, 'rb') as f:
                dados = pickle.load(f)
            
            self.q_table = defaultdict(float, dados['q_table'])
            parametros = dados['parametros']
            self.tamanho_tabuleiro = parametros['tamanho_tabuleiro']
            self.alpha = parametros['alpha']
            self.gamma = parametros['gamma']
            self.epsilon = parametros['epsilon']
            self.epsilon_decay = parametros['epsilon_decay']
            self.epsilon_min = parametros['epsilon_min']
            
            metricas = dados['metricas']
            self.vitorias = metricas['vitorias']
            self.derrotas = metricas['derrotas']
            self.empates = metricas['empates']
            self.historico_epsilon = metricas['historico_epsilon']
            
            print(f"Modelo carregado de {nome_arquivo}")
            print(f"Q-table com {len(self.q_table)} entradas")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
    
    def jogar_contra_humano(self):
        """Permite ao agente jogar contra um humano"""
        jogo = JogoHex(7)
        
        print("=== Q-Learning Agent vs Humano ===")
        prompt = (
                    f"Você quer joga com o {Fore.RED}⬢{Style.RESET_ALL} (X, esquerda-direita) ou {Fore.CYAN}⬢{Style.RESET_ALL} (O, topo-base)?"
                )
        escolha = input(prompt).upper()
        
        if escolha not in ['X', 'O']:
            escolha = 'X'
            print("Escolha inválida, você será X")
        
        self.jogador = 'O' if escolha == 'X' else 'X'
        humano_jogador = escolha
        
        print(f"Você é {humano_jogador}, IA é {self.jogador}")
        jogo.exibir_tabuleiro()
        
        while True:
            if jogo.jogador_atual == humano_jogador:
                print(f"Sua vez (jogador {humano_jogador})!")
                try:
                    linha, coluna = map(int, input("Digite linha e coluna: ").split())
                    if jogo.movimento_valido(linha, coluna):
                        jogo.tabuleiro[linha, coluna] = humano_jogador
                        if jogo.verificar_vitoria(humano_jogador):
                            jogo.exibir_tabuleiro()
                            print("Você venceu!")
                            break
                        jogo.jogador_atual = self.jogador
                    else:
                        print("Movimento inválido!")
                        continue
                except:
                    print("Entrada inválida!")
                    continue
            else:
                print(f"IA ({self.jogador}) está pensando...")
                acao = self.selecionar_acao(jogo, treinamento=False)
                if acao:
                    linha, coluna = acao
                    jogo.tabuleiro[linha, coluna] = self.jogador
                    print(f"IA jogou em {linha}, {coluna}")
                    if jogo.verificar_vitoria(self.jogador):
                        jogo.exibir_tabuleiro()
                        print("IA venceu!")
                        break
                    jogo.jogador_atual = humano_jogador
            
            jogo.exibir_tabuleiro()
            
            if len(self.obter_acoes_possiveis(jogo)) == 0:
                print("Empate!")
                break

def comparar_agentes(qlearning_agent, num_jogos=100, tamanho_tabuleiro=11):
    """
    Compara o desempenho do Q-Learning contra Minimax
    
    Args:
        qlearning_agent: Agente Q-Learning treinado
        num_jogos: Número de jogos para comparação
        tamanho_tabuleiro: Tamanho do tabuleiro
    """
    print(f"\n=== Comparação Q-Learning vs Minimax ({num_jogos} jogos) ===")
    
    vitorias_qlearning = 0
    vitorias_minimax = 0
    empates = 0
    
    for jogo_num in range(num_jogos):
        jogo = JogoHex(tamanho_tabuleiro)
        
        # Alterna quem começa
        if jogo_num % 2 == 0:
            qlearning_agent.jogador = 'X'
            minimax_jogador = 'O'
        else:
            qlearning_agent.jogador = 'O'
            minimax_jogador = 'X'
        
        while True:
            if jogo.jogador_atual == qlearning_agent.jogador:
                # Q-Learning joga
                acao = qlearning_agent.selecionar_acao(jogo, treinamento=False)
                if acao:
                    linha, coluna = acao
                    jogo.tabuleiro[linha, coluna] = qlearning_agent.jogador
                    if jogo.verificar_vitoria(qlearning_agent.jogador):
                        vitorias_qlearning += 1
                        break
                    jogo.jogador_atual = minimax_jogador
            else:
                # Minimax joga
                linha, coluna = jogo.melhor_jogada_ia()
                jogo.tabuleiro[linha, coluna] = minimax_jogador
                if jogo.verificar_vitoria(minimax_jogador):
                    vitorias_minimax += 1
                    break
                jogo.jogador_atual = qlearning_agent.jogador
            
            # Verifica empate
            if len(qlearning_agent.obter_acoes_possiveis(jogo)) == 0:
                empates += 1
                break
        
        if (jogo_num + 1) % 10 == 0:
            print(f"Progresso: {jogo_num + 1}/{num_jogos} jogos")
    
    print(f"\nResultados finais:")
    print(f"Q-Learning: {vitorias_qlearning} vitórias ({vitorias_qlearning/num_jogos*100:.1f}%)")
    print(f"Minimax: {vitorias_minimax} vitórias ({vitorias_minimax/num_jogos*100:.1f}%)")
    print(f"Empates: {empates} ({empates/num_jogos*100:.1f}%)")
    
    return vitorias_qlearning, vitorias_minimax, empates

def plotar_metricas(agente, salvar_grafico=True):
    """Plota métricas de treinamento"""
    if not agente.historico_epsilon:
        print("Nenhum dado de treinamento para plotar")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico do epsilon
    ax1.plot(agente.historico_epsilon)
    ax1.set_title('Decaimento do Epsilon durante o Treinamento')
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Epsilon')
    ax1.grid(True)
    
    # Gráfico de vitórias acumuladas
    episodios = list(range(1, len(agente.historico_epsilon) + 1))
    total_jogos = agente.vitorias + agente.derrotas + agente.empates
    if total_jogos > 0:
        taxa_vitoria = [agente.vitorias / total_jogos * 100] * len(episodios)
        ax2.plot(episodios, taxa_vitoria, label=f'Taxa de Vitórias ({agente.vitorias}/{total_jogos})')
    
    ax2.set_title('Taxa de Vitórias')
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel('Taxa de Vitórias (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if salvar_grafico:
        plt.savefig('qlearning_metricas.png', dpi=300, bbox_inches='tight')
        print("Gráfico salvo como 'qlearning_metricas.png'")
    
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Cria o agente
    agente = QlearningAgent(tamanho_tabuleiro=7, alpha=0.1, gamma=0.9, epsilon=1.0)
    
    print("=== Treinamento do Agente Q-Learning ===")
    escolha = input("1. 🧠 Treinar novo agente\n2. 🤖 Carregar agente existente\n3. ⚔️  Comparar com Minimax\nEscolha: ")
    
    if escolha == '1':
        # Treina o agente
        num_episodios = int(input("Número de episódios de treinamento (padrão 5000): ") or "5000")
        tipo_oponente = input("Tipo de oponente (aleatorio/minimax, padrão aleatorio): ") or "aleatorio"
        
        agente.treinar(num_episodios=num_episodios, oponente=tipo_oponente)
        
        # Plota métricas
        plotar_metricas(agente)
        
        # Opção de jogar contra humano
        jogar = input("Quer jogar contra o agente? (s/n): ").lower()
        if jogar == 's':
            agente.jogar_contra_humano()
    
    elif escolha == '2':
        print("\n=== CARREGAR MODELO EXISTENTE ===")
        print("🌐 Buscando modelos no Google Drive...")
        arquivos_drive = listar_arquivos_drive(PASTA_DRIVE_ID)
        modelos_drive = [
            {"name": f["name"], "id": f["id"], "origem": "drive"}
            for f in arquivos_drive if f["name"].endswith(".pkl")
        ]

        print("📁 Buscando modelos locais...")
        modelos_locais = [
            {"name": f, "origem": "local"}
            for f in os.listdir(CAMINHO_MODELOS) if f.endswith(".pkl")
        ]

        modelos_todos = modelos_locais + modelos_drive

        if not modelos_todos:
            print("❌ Nenhum modelo disponível!")
        else:
            print("\nModelos disponíveis:")
            for i, modelo in enumerate(modelos_todos, 1):
                origem = "[Local]" if modelo["origem"] == "local" else "[Drive]"
                print(f"{i}. {modelo['name']} {origem}")

            try:
                escolha_modelo = int(input("Escolha um modelo (número): ")) - 1
                modelo_escolhido = modelos_todos[escolha_modelo]

                if modelo_escolhido["origem"] == "local":
                    caminho = os.path.join(CAMINHO_MODELOS, modelo_escolhido["name"])
                else:
                    caminho = baixar_arquivo_drive(modelo_escolhido["id"], modelo_escolhido["name"])

                agente.carregar_modelo(caminho)
                agente.jogar_contra_humano()

            except Exception as e:
                print(f"❌ Erro ao carregar modelo: {e}")
    
    elif escolha == '3':
        print("\n=== COMPARAÇÃO Q-LEARNING VS MINIMAX ===")
        print("🌐 Buscando modelos no Google Drive...")
        arquivos_drive = listar_arquivos_drive(PASTA_DRIVE_ID)
        modelos_drive = [
            {"name": f["name"], "id": f["id"], "origem": "drive"}
            for f in arquivos_drive if f["name"].endswith(".pkl")
        ]

        print("📁 Buscando modelos locais...")
        modelos_locais = [
            {"name": f, "origem": "local"}
            for f in os.listdir(CAMINHO_MODELOS) if f.endswith(".pkl")
        ]

        modelos_todos = modelos_locais + modelos_drive

        if not modelos_todos:
            print("❌ Nenhum modelo disponível para comparação!")
        else:
            print("\nModelos disponíveis:")
            for i, modelo in enumerate(modelos_todos, 1):
                origem = "[Local]" if modelo["origem"] == "local" else "[Drive]"
                print(f"{i}. {modelo['name']} {origem}")

            try:
                escolha_modelo = int(input("Escolha um modelo (número): ")) - 1
                modelo_escolhido = modelos_todos[escolha_modelo]

                if modelo_escolhido["origem"] == "local":
                    caminho = os.path.join(CAMINHO_MODELOS, modelo_escolhido["name"])
                else:
                    caminho = baixar_arquivo_drive(modelo_escolhido["id"], modelo_escolhido["name"])

                agente.carregar_modelo(caminho)
                num_jogos = int(input("Número de jogos para comparação (padrão 50): ") or "50")
                comparar_agentes(agente, num_jogos=num_jogos, tamanho_tabuleiro=agente.tamanho_tabuleiro)

            except Exception as e:
                print(f"❌ Erro ao carregar/comparar modelo: {e}")