# EP3-Projeto-Integrador-V-HexGame

# Q-Leaning: Tabuleiro

Neste trabalho, você terá a oportunidade de aprimorar suas habilidades em programação e inteligência artificial, expandindo o jogo de tabuleiro que você já desenvolveu com minimax para um agente que aprende a jogar através do algoritmo Q-Learning. Seu objetivo será treinar o agente para que ele domine o jogo através da experiência e da experimentação, tomando decisões cada vez mais estratégicas.

**Objetivos**

- Implementar o algoritmo Q-Learning no agente do jogo de tabuleiro.
- Treinar o agente para jogar o jogo de forma autônoma.
- Avaliar o desempenho do agente em diferentes cenários.
- Comparar o desempenho do agente com o minimax.

**Desenvolvimento**

1. **Revisão do Minimax:** Comece revisando o código existente do minimax, certificando-se de que compreende completamente seu funcionamento. Este será o ponto de partida para a implementação do Q-Learning.
2. **Implementação do Q-Learning:**
    - Defina o estado do jogo: Represente o estado atual do jogo de forma que o agente possa utilizá-lo para tomar decisões. Isso pode ser feito através de um vetor ou matriz que codifique a posição das peças, o jogador atual, etc.
    - Defina as ações: Identifique todas as ações possíveis que o agente pode realizar em cada estado do jogo, como mover peças, capturar peças, etc.
    - Inicialize a tabela Q: Crie uma tabela Q que armazene os valores Q para cada par estado-ação. Os valores Q representam a recompensa esperada de realizar uma determinada ação em um determinado estado.
    - Atualização da tabela Q: Implemente a regra de atualização do Q-Learning para atualizar os valores Q na tabela após cada jogada. Isso envolve calcular a recompensa recebida e atualizar os valores Q de acordo com a equação de atualização do Q-Learning.
    - Seleção de ação: Defina a estratégia de seleção de ação do agente. Isso pode ser feito de forma exploratória (selecionando ações aleatórias com alguma probabilidade) ou exploratória (selecionando a ação com o maior valor Q).
3. **Função de exploração**
    - Como o agente irá explorar novos estados.
    - Será aleatório ou seguirá uma função?
4. **Treinamento do Agente:**
    - Defina o ambiente de treinamento: Crie um ambiente de treinamento onde o agente possa jogar contra si mesmo ou contra um oponente simples.
    - Execute o treinamento: Treine o agente por um número suficiente de episódios, permitindo que ele explore o jogo e aprenda através da experimentação.
    - Avaliação do desempenho: Monitore o desempenho do agente durante o treinamento, registrando métricas como taxa de vitórias, pontuação média, etc.
    - Deve salvar os dados em um artigo. Sugestão: usar a biblioteca Pickle. Sugestão 2: este arquivo vai ficar gigantesco (1GB) e demora para salvar. Salve depois de N iterações (N > 100, 1000 ou 10000 a depender da velocidade). Salve assincronamente.
5. **Comparação com Minimax:**
    - Jogue partidas entre o agente Q-Learning e o agente minimax.
    - Compare o desempenho dos dois agentes em termos de taxa de vitórias, pontuação média, etc.
    - Analise os resultados e discuta as vantagens e desvantagens de cada abordagem.

**Considerações Adicionais**

- Explore diferentes variantes do algoritmo Q-Learning, como Q-Learning com aprendizagem temporal (TD-learning) ou Q-Learning com elegibilidade.
- Experimente diferentes estratégias de exploração/exploração para a seleção de ação.
- Visualize a tabela Q para observar como os valores Q mudam durante o treinamento.
- Analise os erros cometidos pelo agente durante o treinamento e identifique oportunidades de aprimoramento.

## Entrega

Código fonte e código executado - Exibição do código e detalhamento

Deverá ser entregue o repositório no Github. 

**Não será permitido** de uso de bibliotecas de software de Inteligência Artificial, usar apenas a biblioteca padrão da linguagem.

O que será avaliado:

- Apresentação do tabuleiro, interação com o usuário
- Função de treinamento
- Q-Learning + função de exploração
- Como o agente escolhe o algoritmo para realizar a jogada

O que não será avaliado

- Componentes gráficos. Não me importa se vai ser gráfico 3D ou imprimir no terminal. (Sugestão: imprimir no terminal é trivial)
- Interação com o usuário: não se esperar nenhuma interface homem-máquina sofisticada. Se o usuário digitar qual o seu movimento usando o UCI algébrico, é o suficiente.
