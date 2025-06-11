#!/usr/bin/env python3
"""
Script de Treinamento Intensivo para Agente Hex Q-Learning
Execute este script para treinar automaticamente seu agente com parâmetros otimizados
"""

import os
import time
from datetime import datetime
from qlearning_agent import QlearningAgent, comparar_agentes, plotar_metricas
from utils import listar_arquivos_drive, baixar_arquivo_drive

CAMINHO_MODELOS = "modelos_baixados" # modelos baixados do drive
os.makedirs(CAMINHO_MODELOS, exist_ok=True)

PASTA_DRIVE_ID = "1_PeNiEZy8jhmNWNFVES3bPXU6g8TNzjn"  # ID da pasta do seu Drive

def treinar_agente_intensivo():
    """Treina o agente seguindo uma estratégia progressiva otimizada"""
    
    print("🚀 TREINAMENTO INTENSIVO INICIADO")
    print("=" * 60)
    
    # Parâmetros otimizados
    PARAMETROS = {
        'tamanho_tabuleiro': 7,
        'alpha': 0.15,
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.9995,
        'epsilon_min': 0.05
    }
    
    # Verifica se já existe um modelo em treinamento
    modelo_existente = None
    for arquivo in os.listdir('.'):
        if arquivo.startswith('hex_intensivo_') and arquivo.endswith('.pkl'):
            modelo_existente = arquivo
            break
    
    if modelo_existente:
        print(f"📂 Modelo existente encontrado: {modelo_existente}")
        continuar = input("Deseja continuar o treinamento existente? (s/n): ").lower()
        if continuar == 's':
            agente = QlearningAgent(**PARAMETROS)
            agente.carregar_modelo(modelo_existente)
            print("✅ Modelo carregado com sucesso!")
        else:
            agente = QlearningAgent(**PARAMETROS)
    else:
        agente = QlearningAgent(**PARAMETROS)
    
    # === FASE 1: FUNDAMENTOS ===
    print("\n🎯 FASE 1: TREINAMENTO BÁSICO (vs Aleatório)")
    print("Objetivo: Aprender regras básicas e movimentos fundamentais")
    
    fase1_nome = f"hex_intensivo_fase1_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    
    try:
        agente.treinar(
            num_episodios=25000,
            oponente='aleatorio',
            salvar_a_cada=2500,
            nome_arquivo=fase1_nome
        )
        
        # Avaliação Fase 1
        print("\n📊 AVALIAÇÃO FASE 1:")
        avaliar_agente(agente, "Fase 1")
        
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário")
        agente.salvar_modelo(fase1_nome)
        return agente
    
    # === FASE 2: REFINAMENTO ===
    print("\n🧠 FASE 2: TREINAMENTO AVANÇADO (vs Minimax)")
    print("Objetivo: Desenvolver estratégias mais sofisticadas")
    
    # Ajusta parâmetros para fase avançada
    agente.epsilon = 0.3
    agente.alpha = 0.1
    
    fase2_nome = f"hex_intensivo_fase2_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    
    try:
        agente.treinar(
            num_episodios=20000,
            oponente='minimax',
            salvar_a_cada=2000,
            nome_arquivo=fase2_nome
        )
        
        # Avaliação Fase 2
        print("\n📊 AVALIAÇÃO FASE 2:")
        avaliar_agente(agente, "Fase 2")
        
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário")
        agente.salvar_modelo(fase2_nome)
        return agente
    
    # === FASE 3: POLIMENTO ===
    print("\n✨ FASE 3: POLIMENTO FINAL")
    print("Objetivo: Otimizar estratégias e reduzir inconsistências")
    
    # Parâmetros finais
    agente.epsilon = 0.1
    agente.alpha = 0.05
    
    fase3_nome = f"hex_intensivo_FINAL_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    
    try:
        agente.treinar(
            num_episodios=15000,
            oponente='minimax',
            salvar_a_cada=1500,
            nome_arquivo=fase3_nome
        )
        
        # Avaliação Final
        print("\n🏆 AVALIAÇÃO FINAL:")
        avaliar_agente(agente, "Final")
        
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário")
        agente.salvar_modelo(fase3_nome)
        return agente
    
    # Salva modelo final
    modelo_final = f"hex_PRONTO_PARA_JOGAR_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    agente.salvar_modelo(modelo_final)
    
    print(f"\n🎉 TREINAMENTO COMPLETO!")
    print(f"📁 Modelo final salvo como: {modelo_final}")
    print("🎮 Seu agente está pronto para o desafio!")
    
    return agente

def avaliar_agente(agente, fase):
    """Avalia o desempenho do agente"""
    total_jogos = agente.vitorias + agente.derrotas + agente.empates
    
    if total_jogos > 0:
        taxa_vitoria = agente.vitorias / total_jogos * 100
        print(f"   📈 Taxa de vitórias: {taxa_vitoria:.1f}%")
        print(f"   🎯 Total de jogos: {total_jogos}")
        print(f"   🧠 Tamanho Q-table: {len(agente.q_table)} estados")
        print(f"   🔍 Epsilon atual: {agente.epsilon:.3f}")
    
    # Teste rápido contra Minimax
    print(f"   ⚔️ Testando {fase} contra Minimax (50 jogos)...")
    vit_q, vit_m, emp = comparar_agentes(agente, num_jogos=50, tamanho_tabuleiro=agente.tamanho_tabuleiro)
    taxa_contra_minimax = vit_q / 50 * 100
    print(f"   🏆 Performance vs Minimax: {taxa_contra_minimax:.1f}% vitórias")

def treinar_rapido():
    """Versão rápida para testes"""
    print("⚡ TREINAMENTO RÁPIDO (TESTE)")
    
    agente = QlearningAgent(
        tamanho_tabuleiro=7,
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.1
    )
    
    # Treinamento condensado
    agente.treinar(
        num_episodios=5000,
        oponente='aleatorio',
        salvar_a_cada=1000,
        nome_arquivo='hex_teste_rapido.pkl'
    )
    
    # Teste contra Minimax
    vit_q, vit_m, emp = comparar_agentes(agente, 20, agente.tamanho_tabuleiro)
    print(f"Performance contra Minimax: {vit_q/20*100:.1f}% vitórias")
    
    return agente

def continuar_treinamento_existente():
    """Continua treinamento de um modelo existente"""
    print("🔄 CONTINUANDO TREINAMENTO EXISTENTE")

    # 1. Modelos locais da pasta 'modelos_baixados'
    print("📁 Buscando modelos locais...")
    locais = []
    if os.path.exists('modelos_baixados'):
        locais = [f for f in os.listdir('modelos_baixados') if f.endswith('.pkl')]

    # 2. Modelos no Google Drive
    print("🌐 Buscando modelos no Google Drive...")
    drive_modelos = listar_arquivos_drive(PASTA_DRIVE_ID)

    if not locais and not drive_modelos:
        print("❌ Nenhum modelo encontrado!")
        return

    # 3. Unifica e exibe a lista com origem
    todos_modelos = []
    print("\nModelos disponíveis:")
    idx = 1

    for modelo in locais:
        print(f"{idx}. {modelo} [Local]")
        todos_modelos.append(('local', modelo))
        idx += 1

    for modelo in drive_modelos:
            print(f"{idx}. {modelo['name']} [Drive]")
            todos_modelos.append(('drive', modelo))
            idx += 1

    try:
        escolha = int(input("Escolha um modelo: ")) - 1
        origem, nome = todos_modelos[escolha]
    except:
        print("❌ Escolha inválida!")
        return

    # 4. Lida com o download se for do Drive
    if origem == 'drive':
        nome_drive = nome["name"]
        nome_local = f"modelos_baixados/{nome_drive}"
        print(f"⬇️ Baixando '{nome_drive}' do Google Drive...")
        baixar_arquivo_drive(nome["id"], nome_drive)
    else:
        nome_local = f"modelos_baixados/{nome}"

    # 5. Carrega o agente com o modelo escolhido
    agente = QlearningAgent()
    agente.carregar_modelo(nome_local)

    print(f"Modelo carregado de {nome_local}")
    print(f"Q-table com {len(agente.q_table)} entradas")

    episodios_extras = int(input("Quantos episódios extras? (padrão 10000): ") or "10000")
    oponente = input("Contra quem treinar? (aleatorio/minimax, padrão minimax): ") or "minimax"

    agente.epsilon = max(0.1, agente.epsilon)

    nome_base = nome["name"] if origem == "drive" else nome
    novo_nome = nome_base.replace(".pkl", f"_plus{episodios_extras}.pkl")

    agente.treinar(
        num_episodios=episodios_extras,
        oponente=oponente,
        salvar_a_cada=max(1000, episodios_extras // 10),
        nome_arquivo=novo_nome
    )

    print(f"✅ Treinamento adicional concluído! Modelo salvo como: {novo_nome}")
    return agente

if __name__ == "__main__":
    print("🎯 SISTEMA DE TREINAMENTO INTENSIVO - HEX Q-LEARNING 🎯 ")
    print("=" * 60)
    print("1. 🚀 Treinamento Intensivo Completo (~60k episódios)")
    print("2. ⚡ Treinamento Rápido para Teste (5k episódios)")
    print("3. 🔄 Continuar Treinamento Existente")
    print("4. 🎮 Testar Modelo Existente")
    
    opcao = input("\nEscolha uma opção: ").strip()
    
    if opcao == "1":
        agente = treinar_agente_intensivo()
        
        # Opção de jogar imediatamente
        jogar = input("\n🎮 Quer jogar contra o agente agora? (s/n): ").lower()
        if jogar == 's':
            agente.jogar_contra_humano()
            
    elif opcao == "2":
        agente = treinar_rapido()
        
        jogar = input("\n🎮 Quer jogar contra o agente? (s/n): ").lower()
        if jogar == 's':
            agente.jogar_contra_humano()
            
    elif opcao == "3":
        continuar_treinamento_existente()
        
    elif opcao == "4":
        # Testar modelo existente
        print("🎮 TESTAR MODELO EXISTENTE")

        # Modelos locais
        print("📁 Buscando modelos locais...")
        locais = []
        if os.path.exists('modelos_baixados'):
            locais = [f for f in os.listdir('modelos_baixados') if f.endswith('.pkl')]

        # Modelos do Google Drive
        print("🌐 Buscando modelos no Google Drive...")
        drive_modelos = listar_arquivos_drive(PASTA_DRIVE_ID)

        if not locais and not drive_modelos:
            print("❌ Nenhum modelo encontrado!")
        else:
            todos_modelos = []
            print("\nModelos disponíveis:")
            idx = 1

            for modelo in locais:
                print(f"{idx}. {modelo} [Local]")
                todos_modelos.append(('local', modelo))
                idx += 1

            for modelo in drive_modelos:
                print(f"{idx}. {modelo['name']} [Drive]")
                todos_modelos.append(('drive', modelo))
                idx += 1

            try:
                escolha = int(input("Escolha um modelo: ")) - 1
                origem, nome = todos_modelos[escolha]
            except:
                print("❌ Escolha inválida!")
            else:
                if origem == 'drive':
                    nome_drive = nome["name"]
                    print(f"⬇️ Baixando '{nome_drive}' do Google Drive...")
                    baixar_arquivo_drive(nome["id"], nome_drive)
                    nome_local = f"modelos_baixados/{nome_drive}"
                else:
                    nome_local = f"modelos_baixados/{nome}"

                agente = QlearningAgent()
                agente.carregar_modelo(nome_local)

                print("🔬 Testando modelo...")
                avaliar_agente(agente, "Teste")

                jogar = input("\n🎮 Quer jogar contra ele? (s/n): ").lower()
                if jogar == 's':
                    agente.jogar_contra_humano()
