import numpy as np
import heapq
from colorama import init, Fore, Style
init(autoreset=True)

class JogoHex:
    def __init__(self, tamanho=11):
        self.tamanho = tamanho
        self.tabuleiro = np.full((tamanho, tamanho), '⬡', dtype=str)
        self.jogador_atual = 'X'
        self.modo_jogo = None
        self.dificuldade = 2
        # Variáveis para a regra de swap (pie rule)
        self.swap_available = True
        self.first_move = None  # guarda a posição do primeiro movimento

    def exibir_tabuleiro(self):
        # Cabeçalho
        print("\n==========================================")
        print(f"{Fore.RED}⬢{Style.RESET_ALL} vence conectando esquerda ↔ direita")
        print(f"{Fore.CYAN}⬢{Style.RESET_ALL} vence conectando topo ↓ base")
        print("==========================================\n")
        print("   " + "   ".join(f"{i}" for i in range(self.tamanho)))
        # Linhas
        for linha in range(self.tamanho):
            espacamento = " " * (2 * linha)
            glyphs = []
            for j in range(self.tamanho):
                val = self.tabuleiro[linha, j]
                if val == 'X':
                    # Hexágono cheio vermelho
                    glyph = Fore.RED   + '⬢' + Style.RESET_ALL
                elif val == 'O':
                    # Hexágono cheio azul
                    glyph = Fore.CYAN  + '⬢' + Style.RESET_ALL
                else:
                    # Hexágono contornado (vazio)
                    glyph = '⬡'
                glyphs.append(glyph)
            linha_tabuleiro = "   ".join(glyphs)
            print(f"{espacamento}{linha:2} {linha_tabuleiro}  {linha}")
        # Rodapé alinhado sob a última linha
        n = self.tamanho
        prefixo_rodape = " " * (2*(n-1) + 3 + 1)
        print(prefixo_rodape + "   ".join(f"{i}" for i in range(n)) + "\n")

    def movimento_valido(self, linha, coluna):
        return 0 <= linha < self.tamanho and 0 <= coluna < self.tamanho and self.tabuleiro[linha, coluna] == '⬡'

    def fazer_movimento(self, linha, coluna):
        if not self.movimento_valido(linha, coluna):
            return False
        self.tabuleiro[linha, coluna] = self.jogador_atual
        if self.first_move is None:
            self.first_move = (linha, coluna)
        if self.verificar_vitoria(self.jogador_atual):
            print(f"🎉 Jogador {self.jogador_atual} venceu! 🎉")
            return True
        self.jogador_atual = 'O' if self.jogador_atual == 'X' else 'X'
        return True

    def verificar_vitoria(self, jogador):
        visitados = set()
        direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

        def bfs(linha, coluna):
            if (linha, coluna) in visitados:
                return False
            if jogador == 'X' and coluna == self.tamanho - 1:
                return True
            if jogador == 'O' and linha == self.tamanho - 1:
                return True
            visitados.add((linha, coluna))
            for deslocamento_linha, deslocamento_coluna in direcoes:
                nova_linha, nova_coluna = linha + deslocamento_linha, coluna + deslocamento_coluna
                if 0 <= nova_linha < self.tamanho and 0 <= nova_coluna < self.tamanho and self.tabuleiro[nova_linha, nova_coluna] == jogador:
                    if bfs(nova_linha, nova_coluna):
                        return True
            return False

        posicoes_iniciais = [(i, 0) for i in range(self.tamanho) if self.tabuleiro[i, 0] == 'X'] if jogador == 'X' else [(0, i) for i in range(self.tamanho) if self.tabuleiro[0, i] == 'O']
        return any(bfs(linha, coluna) for linha, coluna in posicoes_iniciais)

    def calcular_utilidade(self, jogador):
        oponente = 'O' if jogador == 'X' else 'X'
        return self.heuristica_caminho_minimo(jogador) - self.heuristica_caminho_minimo(oponente)

    def heuristica_caminho_minimo(self, jogador):
        tamanho = self.tamanho
        direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        visitado = np.full((tamanho, tamanho), False)
        heap = []

        if jogador == 'X':
            for i in range(tamanho):
                if self.tabuleiro[i, 0] != 'O':
                    heapq.heappush(heap, (0 if self.tabuleiro[i, 0] == jogador else 1, i, 0))
        else:
            for j in range(tamanho):
                if self.tabuleiro[0, j] != 'X':
                    heapq.heappush(heap, (0 if self.tabuleiro[0, j] == jogador else 1, 0, j))

        while heap:
            custo, linha, coluna = heapq.heappop(heap)
            if visitado[linha, coluna]:
                continue
            visitado[linha, coluna] = True

            if (jogador == 'X' and coluna == tamanho - 1) or (jogador == 'O' and linha == tamanho - 1):
                return tamanho - custo

            for dl, dc in direcoes:
                nl, nc = linha + dl, coluna + dc
                if 0 <= nl < tamanho and 0 <= nc < tamanho and not visitado[nl, nc]:
                    if self.tabuleiro[nl, nc] == jogador:
                        heapq.heappush(heap, (custo, nl, nc))
                    elif self.tabuleiro[nl, nc] == '⬡':
                        heapq.heappush(heap, (custo + 1, nl, nc))
        return 0

    def forca_conectada(self, linha, coluna, jogador):
        return sum(1 for deslocamento_linha, deslocamento_coluna in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)] if 0 <= linha + deslocamento_linha < self.tamanho and 0 <= coluna + deslocamento_coluna < self.tamanho and self.tabuleiro[linha + deslocamento_linha, coluna + deslocamento_coluna] == jogador) + 1

    def gerar_sucessores(self):
        sucessores = []
        for linha in range(self.tamanho):
            for coluna in range(self.tamanho):
                if self.tabuleiro[linha, coluna] == '⬡':
                    novo_tabuleiro = self.tabuleiro.copy()
                    novo_tabuleiro[linha, coluna] = self.jogador_atual
                    sucessores.append((linha, coluna, novo_tabuleiro))
        return sucessores
    
    def clonar(self):
        """Cria uma cópia do estado atual do jogo para simulação no minimax"""
        clone = JogoHex(self.tamanho)
        clone.tabuleiro = self.tabuleiro.copy()
        clone.jogador_atual = self.jogador_atual
        return clone
    
    def minimax(self, profundidade, eh_maximizando, jogador, alfa=float('-inf'), beta=float('inf')):
        """
        Implementação do algoritmo Minimax com poda Alfa-Beta
        
        Args:
            profundidade: Níveis restantes para explorar na árvore
            eh_maximizando: Verdadeiro se for o turno do jogador maximizador
            jogador: 'X' ou 'O', o jogador para quem estamos calculando a utilidade
            alfa, beta: Valores para poda alfa-beta
            
        Returns:
            Melhor valor de utilidade encontrado neste nível
        """
        # Casos base: jogo acabou ou profundidade máxima atingida
        if profundidade == 0 or self.verificar_vitoria('X') or self.verificar_vitoria('O'):
            return self.calcular_utilidade(jogador)
        
        # Identificar o jogador atual na simulação
        jogador_atual = jogador if eh_maximizando else ('O' if jogador == 'X' else 'X')
        oponente = 'O' if jogador_atual == 'X' else 'X'
        
        # Simulamos todos os movimentos possíveis
        movimentos_vazios = [(i, j) for i in range(self.tamanho) for j in range(self.tamanho) if self.tabuleiro[i, j] == '⬡']
        
        # Para reduzir o espaço de busca em tabuleiros grandes, priorizamos células próximas às existentes
        if len(movimentos_vazios) > 30:
            movimentos_prioritarios = []
            direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]
            
            for linha, coluna in movimentos_vazios:
                # Verifica se há peças adjacentes
                for dl, dc in direcoes:
                    nova_l, nova_c = linha + dl, coluna + dc
                    if (0 <= nova_l < self.tamanho and 0 <= nova_c < self.tamanho and 
                        self.tabuleiro[nova_l, nova_c] != '⬡'):
                        movimentos_prioritarios.append((linha, coluna))
                        break
            
            # Se encontramos movimentos prioritários, use-os
            if movimentos_prioritarios:
                movimentos_vazios = movimentos_prioritarios
        
        if eh_maximizando:
            melhor_valor = float('-inf')
            for linha, coluna in movimentos_vazios:
                # Simula o movimento
                jogo_simulado = self.clonar()
                jogo_simulado.tabuleiro[linha, coluna] = jogador_atual
                jogo_simulado.jogador_atual = oponente
                
                # Avalia com minimax recursivamente
                valor = jogo_simulado.minimax(profundidade - 1, False, jogador, alfa, beta)
                melhor_valor = max(melhor_valor, valor)
                
                # Atualiza alfa para poda
                alfa = max(alfa, melhor_valor)
                if beta <= alfa:
                    break  # Poda beta
                    
            return melhor_valor
        else:
            pior_valor = float('inf')
            for linha, coluna in movimentos_vazios:
                # Simula o movimento
                jogo_simulado = self.clonar()
                jogo_simulado.tabuleiro[linha, coluna] = jogador_atual
                jogo_simulado.jogador_atual = oponente
                
                # Avalia com minimax recursivamente
                valor = jogo_simulado.minimax(profundidade - 1, True, jogador, alfa, beta)
                pior_valor = min(pior_valor, valor)
                
                # Atualiza beta para poda
                beta = min(beta, pior_valor)
                if beta <= alfa:
                    break  # Poda alfa
                    
            return pior_valor
    
    def melhor_jogada_ia(self):
        """Determina a melhor jogada para a IA usando o algoritmo Minimax"""
        melhor_valor = float('-inf')
        melhores_jogadas = []
        
        # Caso especial: primeira jogada no centro ou próxima ao centro
        celulas_vazias = np.sum(self.tabuleiro == '⬡')
        if celulas_vazias == self.tamanho * self.tamanho or celulas_vazias == self.tamanho * self.tamanho - 1:
            centro = self.tamanho // 2
            return centro, centro
        
        # Ajusta a profundidade com base no tamanho do tabuleiro e na fase do jogo
        profundidade_maxima = self.dificuldade
        if self.tamanho > 7:
            # Reduz a profundidade para tabuleiros grandes
            profundidade_maxima = min(self.dificuldade, 2)
        
        # Encontrar todas as jogadas possíveis
        movimentos_vazios = [(i, j) for i in range(self.tamanho) for j in range(self.tamanho) if self.tabuleiro[i, j] == '⬡']
        
        # Para reduzir o espaço de busca em tabuleiros grandes, priorizamos células próximas às existentes
        if len(movimentos_vazios) > 30:
            movimentos_prioritarios = []
            direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]
            
            for linha, coluna in movimentos_vazios:
                # Verifica se há peças adjacentes
                for dl, dc in direcoes:
                    nova_l, nova_c = linha + dl, coluna + dc
                    if (0 <= nova_l < self.tamanho and 0 <= nova_c < self.tamanho and 
                        self.tabuleiro[nova_l, nova_c] != '⬡'):
                        movimentos_prioritarios.append((linha, coluna))
                        break
            
            # Se encontramos movimentos prioritários, use-os
            if movimentos_prioritarios:
                movimentos_vazios = movimentos_prioritarios
                
        print(f"IA está pensando... (avaliando {len(movimentos_vazios)} jogadas possíveis)")
                
        for linha, coluna in movimentos_vazios:
            # Simula o movimento
            jogo_simulado = self.clonar()
            jogo_simulado.tabuleiro[linha, coluna] = self.jogador_atual
            jogo_simulado.jogador_atual = 'O' if self.jogador_atual == 'X' else 'X'
            
            # Avalia com minimax
            valor = jogo_simulado.minimax(profundidade_maxima - 1, False, self.jogador_atual)
            
            # Atualiza a melhor jogada
            if valor > melhor_valor:
                melhor_valor = valor
                melhores_jogadas = [(linha, coluna)]
            elif valor == melhor_valor:
                melhores_jogadas.append((linha, coluna))
        
        # Se houver múltiplas jogadas com o mesmo valor, escolhe uma aleatoriamente
        if melhores_jogadas:
            return melhores_jogadas[np.random.randint(0, len(melhores_jogadas))]
        
        # Escolhe uma jogada aleatória se algo der errado
        vazios = np.where(self.tabuleiro == '⬡')
        indice = np.random.randint(0, len(vazios[0]))
        return vazios[0][indice], vazios[1][indice]

    def jogar(self):
        swap_available = True
        move_count = 0
        first_move = None

        # escolher modo HH ou HI
        while self.modo_jogo not in ['HH', 'HI']:
            self.modo_jogo = input("Escolha o modo de jogo (HH para Humano vs Humano, HI para Humano vs IA): ").upper()

        # no modo HI, pergunta quem será X ou O e a dificuldade
        if self.modo_jogo == 'HI':
            humano_jogador = None
            while humano_jogador not in ['X', 'O']:
                prompt = (
                    f"Você quer jogar com o {Fore.RED}⬢{Style.RESET_ALL} (X) ou {Fore.CYAN}⬢{Style.RESET_ALL} (O)? "
                )
                humano_jogador = input(prompt).upper()
            ia_jogador = 'O' if humano_jogador == 'X' else 'X'

            while True:
                try:
                    dif = int(input("Escolha a dificuldade (1-fácil, 2-médio, 3-difícil): "))
                    if 1 <= dif <= 3:
                        self.dificuldade = dif
                        break
                except ValueError:
                    pass

        # exibe o tabuleiro inicial
        self.exibir_tabuleiro()

        while True:
            print(f"Jogador {self.jogador_atual}, sua vez!")
            print(f"Avaliando o tabuleiro: {self.calcular_utilidade(self.jogador_atual)} pontos")

            # turno do X sempre lê coordenadas do usuário
            if self.jogador_atual == 'X':
                sucessores = self.gerar_sucessores()
                print(f"Total de sucessores possíveis: {len(sucessores)}")
                try:
                    linha, coluna = map(int, input("Digite linha e coluna (ex: 3 4): ").split())
                except ValueError:
                    print("Entrada inválida! Use números separados por espaço.")
                    continue
                if not self.fazer_movimento(linha, coluna):
                    print("Movimento inválido!")
                    continue
                if first_move is None:
                    first_move = (linha, coluna)
                move_count += 1
                self.exibir_tabuleiro()

            # turno do O no modo HH (humano)
            elif self.modo_jogo == 'HH':
                # oferta de swap apenas na primeira jogada de O
                if move_count == 1 and swap_available:
                    swap_available = False
                    resp = input("Deseja trocar a primeira peça do adversário por um hexágono da sua cor? (s/n): ").strip().lower()
                    if resp == 's':
                        i, j = first_move
                        self.tabuleiro[i, j] = 'O'
                        print("🔄 Jogador O realizou swap!")
                        self.jogador_atual = 'X'
                        move_count += 1
                        self.exibir_tabuleiro()
                        continue
                # se não swap, lê normalmente
                sucessores = self.gerar_sucessores()
                print(f"Total de sucessores possíveis: {len(sucessores)}")
                try:
                    linha, coluna = map(int, input("Digite linha e coluna (ex: 3 4): ").split())
                except ValueError:
                    print("Entrada inválida! Use números separados por espaço.")
                    continue
                if not self.fazer_movimento(linha, coluna):
                    print("Movimento inválido!")
                    continue
                move_count += 1
                self.exibir_tabuleiro()

            # turno do O no modo HI (IA)
            else:
                # oferta de swap estratégico
                if move_count == 1 and swap_available:
                    swap_available = False
                    i, j = first_move
                    util_before = self.calcular_utilidade('O')
                    # simula troca
                    old = self.tabuleiro[i, j]
                    self.tabuleiro[i, j] = 'O'
                    util_after = self.calcular_utilidade('O')
                    self.tabuleiro[i, j] = old
                    if util_after > util_before:
                        self.tabuleiro[i, j] = 'O'
                        print("🔄 IA realizou swap estratégico!")
                        self.jogador_atual = 'X'
                        move_count += 1
                        self.exibir_tabuleiro()
                        continue
                # jogada normal da IA
                print(f"IA ({ia_jogador}) está jogando...")
                linha, coluna = self.melhor_jogada_ia()
                print(f"IA jogou em {linha}, {coluna}")
                if not self.fazer_movimento(linha, coluna):
                    print("Movimento inválido da IA!")
                    continue
                move_count += 1
                self.exibir_tabuleiro()

            # verifica vitória
            if self.verificar_vitoria('X') or self.verificar_vitoria('O'):
                break
