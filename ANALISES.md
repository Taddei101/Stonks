# Análises de risco e rebalanceamento — plano de implementação

Documento de escopo para implementação incremental. Cada tarefa tem objetivo,
encaixe na arquitetura atual e critério de pronto.

## Situação

- **Fase 0 — FEITA.** `requirements.txt` em UTF-8 com as diretas separadas,
  `precos.historico_total()` com cache que distingue as duas séries,
  `carteira/series.py`, `config/blocos.json` e `carteira/blocos.py`.
- **Fase 1 — FEITA.** `carteira/risco.py` com Ledoit-Wolf,
  `carteira/componentes.py` com apostas efetivas, contribuição de risco por
  ativo e por bloco, `carteira/relatorio.py` e o comando `main.py --risco`.
- **Fase 2 — FEITA.** `carteira/alocacoes.py` com 1/N, inverso da
  volatilidade, paridade de risco e mínima variância; `carteira/cdi.py` com a
  série do Banco Central; `carteira/comparar.py` com intervalo de confiança
  por reamostragem em blocos; comando `main.py --comparar`.
- **Fase 3 — adiada.** Com 5,01 apostas efetivas medidas, a hipótese de
  concentração perdeu força e os cenários de estresse deixaram de ser
  prioridade.
- **Validação deslizante — FEITA.** `carteira/validacao.py` com janela
  rolante e ancorada, medindo giro e sem vazamento de futuro. Três regras
  novas que usam retorno (`markowitz`, `markowitz_encolhido`, `tendencia`)
  entraram para serem testadas; `momentum_12_1` entrou em 05/11 (ver
  "Regras de alocação e referências"). Comando `main.py --validar`.
- **Fase 4 — pendente.** Bandas de rebalanceamento, proventos como
  instrumento, custos e imposto. É o que responde "quando comprar ou vender".
  O custo de transição é o item que pode inverter a decisão de mudar o alvo.

Extras não previstos no plano: extrato sintético em `tests/dados/`, suíte de
185 testes que antes não rodava em clone limpo, correção do mês em andamento
na série mensal, e redução do `blocos.json` a uma dimensão depois que os
componentes principais se mostraram melhores que o agrupamento manual.

**Comandos:**

    python main.py --risco --alvo --janela 3
    python main.py --comparar --janela 3
    python main.py --validar --janela 10
    python main.py --validar --janela 10 --modo ancorada
    python main.py --comparar --tickers PETR4,VALE3,WEGE3,IVVB11

Rode os dois em `--janela 1`, `3` e `5`. Se o diagnóstico mudar muito entre
as janelas, isso sozinho já diz que a estimativa não é estável o bastante
para decidir alocação.

No relatório de risco, olhe a intensidade de encolhimento: ela mede quanto a
matriz amostral precisou ser corrigida. Na comparação, olhe primeiro a
largura do intervalo de Sharpe — diferenças menores que ela não significam
nada.

---

**Princípio que atravessa tudo:** nenhuma função de análise pode olhar dados
posteriores à data que está avaliando. Se qualquer passo usa o conjunto
completo para decidir algo em `t`, a validação não vale nada.

**Princípio de arquitetura:** os módulos novos seguem o padrão já estabelecido
em `retorno.py` e `rebalance.py` — funções puras, sem rede, sem matplotlib.
Rede fica em `precos.py`, gráfico fica em `graficos.py`.

---

## Fase 0 — Fundação

Nada da Fase 1 em diante funciona sem isto.

### 0.1 — Corrigir o `requirements.txt`

**Problema:** o arquivo está codificado em UTF-16 com BOM. Isso acontece quando
se roda `pip freeze > requirements.txt` no PowerShell, que usa UTF-16 por
padrão na redireção. O `pip install -r requirements.txt` falha num clone limpo.

**Fazer:** regravar em UTF-8. No PowerShell:

    pip freeze | Out-File -Encoding utf8 requirements.txt

Separar dependências diretas das transitivas. Hoje o arquivo lista 38 pacotes,
mas as diretas são poucas: `pandas`, `numpy`, `matplotlib`, `yfinance`,
`openpyxl`, `pyarrow`, `pytest`. O resto veio junto.

**Pronto quando:** `pip install -r requirements.txt` funciona num ambiente
virtual novo.

### 0.2 — Série de retorno total (`carteira/series.py`, novo)

**Problema:** `precos._baixar` usa `auto_adjust=False`, o que é a decisão certa
para valor de mercado da carteira — e está bem justificada no docstring. Mas
para correlação, covariância e qualquer medida de risco a série precisa ser de
retorno total: com proventos reinvestidos e ajustada por desdobramento. Sem
isso, toda ação que paga muito dividendo aparece com retorno subestimado e com
um salto artificial para baixo na data-ex, o que contamina a matriz de
covariância.

**Fazer:** nova função em `precos.py` que baixa com `auto_adjust=True`,
mantida separada de `historico()`. Nome sugerido: `historico_total()`. Cache
em parquet separado, porque as duas séries não são intercambiáveis.

Em `carteira/series.py`, novo módulo com funções puras:

- `retornos_diarios(precos) -> DataFrame` — retorno logarítmico ou simples por
  ticker. Escolher um e documentar: simples para agregação entre ativos,
  logarítmico para agregação no tempo. Para covariância, simples.
- `retornos_mensais(precos) -> DataFrame` — reamostragem mensal. A maior parte
  das análises de risco roda em frequência mensal; diário tem ruído de
  microestrutura e assincronia de negociação que distorce correlação.
- `retorno_carteira(retornos, pesos) -> Series` — série da carteira a pesos
  fixos, para comparar alternativas de alocação sem precisar simular negócios.

**Cuidado:** IVVB11 só tem histórico a partir de 2014 e TAEE11 tem liquidez
irregular em períodos antigos. Decidir e documentar a política para janela em
que algum ativo não existe: cortar o período comum, ou tratar por ativo. Cortar
é mais honesto e mais simples.

**Pronto quando:** existe teste que confere que o retorno total de um ativo com
dividendo conhecido bate com cálculo manual.

### 0.3 — Blocos de risco (`config/blocos.json`, novo)

**Objetivo:** tornar explícito o agrupamento que hoje só existe implicitamente.

Mapeamento de cada ticker para os fatores que o afetam. Um ativo pertence a
mais de um bloco — isso é proposital:

    {
      "setor":   {"ITUB4": "bancos", "BBAS3": "bancos",
                  "VALE3": "commodities", "PETR4": "commodities",
                  "CPLE3": "eletricas", "TAEE11": "eletricas",
                  "IVVB11": "exterior"},
      "controle":{"PETR4": "estatal", "BBAS3": "estatal", "CPLE3": "estatal",
                  "ITUB4": "privado", "VALE3": "privado",
                  "TAEE11": "privado", "IVVB11": "privado"},
      "moeda":   {"IVVB11": "USD", "...": "BRL"}
    }

**Por que importa:** com o alvo atual, bancos somam 35,5%, commodities 32,2%,
elétricas 30,6% e exterior 1,6%. Controle estatal soma 40,4%. Risco político e
regulatório é comum a PETR4, BBAS3 e CPLE3 e fica quieto por anos antes de se
manifestar de uma vez — a correlação histórica de 3 anos subestima isso.

**Pronto quando:** `carregar_blocos()` lê o arquivo e valida que todo ticker do
`alvo.json` está mapeado em todas as dimensões.

---

## Fase 1 — Diagnóstico da estrutura de risco

Esta é a fase que responde "vale a pena diversificar". Mede estrutura, não
desempenho passado — estrutura tem relação muito melhor com o futuro.

### 1.1 — Matriz de correlação e covariância (`carteira/risco.py`, novo)

- `matriz_correlacao(retornos) -> DataFrame`
- `matriz_covariancia(retornos, metodo) -> DataFrame` com `metodo` em
  `{"amostral", "ledoit_wolf"}`

**Por que encolhimento (Ledoit-Wolf):** com 7 ativos e 36 observações mensais a
covariância amostral é mal condicionada — os autovalores extremos estão muito
errados, e é exatamente neles que o otimizador se apoia. O encolhimento puxa a
matriz na direção de um alvo estruturado e reduz drasticamente o erro.
Implementável em ~30 linhas sem adicionar `scikit-learn`, mas se preferir a
dependência, `sklearn.covariance.LedoitWolf` resolve.

**Pronto quando:** a matriz com encolhimento é sempre positiva definida, mesmo
com janela curta. Incluir teste com janela de 12 meses.

### 1.2 — Número efetivo de apostas

**Objetivo:** responder quantas apostas independentes a carteira realmente tem.
Você tem 7 papéis; a suspeita é que sejam ~3 apostas.

**Fazer:** decomposição em componentes principais da matriz de correlação.
Reportar quantos componentes explicam 90% da variância, e a entropia dos
autovalores normalizados (número efetivo de apostas = exponencial da entropia).

**Interpretação:** se 2 componentes explicam 85%+, a diversificação nominal é
ilusória. Nesse caso identificar o que cada componente representa olhando as
cargas — o primeiro quase certamente é "risco Brasil", o segundo talvez
commodities contra doméstico.

**Pronto quando:** sai uma tabela com autovalor, variância explicada acumulada e
as cargas de cada ativo nos 3 primeiros componentes.

### 1.3 — Contribuição de risco por ativo e por bloco

**Objetivo:** separar concentração de capital de concentração de risco. São
coisas diferentes e a segunda é a que importa.

**Fazer:**

- Contribuição marginal ao risco: `(Σw)ᵢ / σ_carteira`
- Contribuição total: `wᵢ × (Σw)ᵢ / σ_carteira`, que soma à volatilidade total
- Agregar por bloco usando `config/blocos.json`

**Pronto quando:** tabela com peso de capital e peso de risco lado a lado, por
ativo e por bloco. A expectativa é ver VALE3 e PETR4 com participação no risco
bem maior que no capital.

### 1.4 — Relatório de estrutura

Juntar 1.1 a 1.3 num comando só: `python main.py --risco`.

Deve responder em uma tela: quantas apostas efetivas, quais blocos dominam o
risco, quais pares têm correlação acima de 0,7, e quanto do risco vem do fator
comum estatal.

---

## Fase 2 — Comparação com referências

Atenção: esta fase é informativa, não decisória. Com ~36 observações mensais, a
margem de erro supera qualquer diferença plausível entre alternativas. Serve
para detectar erro grosseiro, não para escolher alocação.

### 2.1 — Carteiras de referência (`carteira/benchmark.py`, novo)

Construir, sobre o mesmo período e os mesmos ativos:

- **1/N** — pesos iguais, o padrão difícil de superar
- **Inverso da volatilidade** — `wᵢ ∝ 1/σᵢ`, não precisa estimar retorno
- **Paridade de risco** — contribuição de risco igual entre ativos
- **Mínima variância com encolhimento** — usando 1.1
- **Equilíbrio por bloco** — peso igual entre bancos, commodities, elétricas,
  exterior; dentro do bloco, igual entre os papéis
- **Alvo atual** — o `alvo.json`

E referências externas: Ibovespa (`^BVSP`) e CDI. O CDI não vem do yfinance;
usar a série 12 do SGS do Banco Central, com cache local.

**Pronto quando:** todas devolvem série de retorno mensal no mesmo índice.

### 2.2 — Comparação com intervalo de confiança

**Fazer:** para cada carteira de referência, calcular retorno anualizado,
volatilidade, Sharpe (sobre CDI), perda máxima e perda média dos 5% piores
meses.

**O ponto central:** acompanhar cada Sharpe de um intervalo de confiança obtido
por reamostragem. Sem isso a tabela sugere precisão que não existe.

**Pronto quando:** o relatório afirma explicitamente se a diferença entre o alvo
atual e 1/N é ou não estatisticamente distinguível. A resposta esperada é que
não seja — e saber disso é o resultado útil.

### 2.3 — Decomposição TWR contra MWR

Você já calcula os dois. Falta explicitar a diferença como uma métrica própria:
o custo (ou ganho) do momento dos seus aportes.

**Pronto quando:** o relatório mostra os dois lado a lado e a diferença em
pontos percentuais ao ano, com a observação de que só o TWR é comparável com
referência externa.

---

## Fase 3 — Cauda e estresse

Aqui mora o custo real da concentração. Ele não aparece na média; aparece de
uma vez.

### 3.1 — Perda máxima e análise de cauda

Por carteira de referência: perda máxima no período, duração da perda, tempo até
recuperar, e perda média dos 5% piores meses.

Reportar também os 10 piores meses da carteira atual com o que aconteceu no
mercado naquele mês — serve para reconhecer qual fator machuca.

### 3.2 — Cenários de estresse

**Objetivo:** dimensionar o estrago sem precisar estimar probabilidade.

Cenários a modelar, cada um como choque simultâneo nos ativos afetados:

- Intervenção política em estatal (PETR4, BBAS3, CPLE3 juntos)
- Queda forte do minério de ferro (VALE3, com arraste em PETR4)
- Mudança regulatória no setor elétrico (CPLE3, TAEE11)
- Salto do dólar (IVVB11 sobe, Brasil cai)
- Crise bancária doméstica (ITUB4, BBAS3)

Calibrar a magnitude pelos piores episódios históricos de cada tipo, não por
chute. Reportar o impacto no alvo atual e em cada alternativa.

**Pronto quando:** sai uma tabela cenário × carteira com a perda estimada. Esta
tabela é o insumo principal da decisão de diversificar.

### 3.3 — Reamostragem em blocos

**Objetivo:** gerar cenários futuros preservando as características que a
distribuição normal apaga — agrupamento de volatilidade e caudas pesadas.

**Fazer:** reamostragem em blocos dos retornos mensais (blocos de 3 a 6 meses,
sorteados com reposição) para gerar milhares de trajetórias. Para cada carteira
de referência, reportar a distribuição do retorno acumulado em 5 e 10 anos, com
percentis 5, 25, 50, 75 e 95.

**Não usar** Monte Carlo com normal multivariada. Ela subestima sistematicamente
exatamente o risco que interessa aqui.

**Pronto quando:** a distribuição simulada reproduz a assimetria e a curtose da
série histórica dentro de tolerância razoável. Incluir esse teste.

---

## Fase 4 — Política de rebalanceamento

Só faz sentido depois que a Fase 3 disser se a alocação muda.

### 4.1 — Bandas de tolerância (estender `rebalance.py`)

**Hoje:** o `plano_aporte` distribui o aporte proporcionalmente ao déficit de
todos os ativos defasados. Funciona, mas dispara movimento mesmo para desvios
minúsculos.

**Fazer:** banda relativa, não absoluta. Com alvo de 14%, uma banda de ±5 pontos
percentuais praticamente nunca dispara. Usar percentual do próprio alvo — 25% de
14% dá ±3,5 pontos. Para IVVB11 a 1,6% a banda absoluta é inútil; a relativa
funciona.

Acrescentar ao `alvo.json` ou a um `config/politica.json`:

    {"banda_relativa": 0.25, "banda_minima_pp": 1.0, "ordem": ["proventos", "aporte", "venda"]}

Manter a filosofia atual: vender é último recurso.

### 4.2 — Fluxo de proventos como instrumento

**Objetivo:** você já separa proventos de aporte externo no `caixa.py`. Falta
usar essa separação no rebalanceamento.

A carteira é pesada em pagadores — BBAS3, PETR4, TAEE11 e CPLE3. Direcionar
proventos para quem está abaixo do alvo rebalanceia sem corretagem e sem fato
gerador. A ordem correta é: primeiro proventos acumulados no caixa, depois
aporte externo, e só então venda.

**Pronto quando:** o plano mostra separadamente quanto do rebalanceamento sai de
provento e quanto de aporte novo.

### 4.3 — Custos e tributação na simulação (`carteira/custos.py`, novo)

Sem isto, qualquer simulação de política superestima o benefício de rebalancear.

Modelar:

- Corretagem e emolumentos por ordem
- Diferença entre compra e venda, maior em TAEE11 e CPLE3 que em ITUB4
- **Imposto de renda sobre venda com lucro.** Em 2026 segue valendo a isenção
  para vendas de ações no mercado à vista até R$ 20 mil por mês. Modelar isso
  explicitamente: fatiar vendas de rebalanceamento para caber no limite elimina
  o custo tributário.
- **IVVB11 não tem essa isenção.** Fundo de índice paga 15% sobre o ganho
  independentemente do valor vendido. Consequência prática: rebalancear pelas
  ações sempre que possível e mexer no IVVB11 o mínimo.

O cálculo precisa de preço médio por ativo, que sai do ledger em `posicao.py`.

**Pronto quando:** dado um plano de venda, a função devolve corretagem, imposto
e líquido, com o fatiamento por mês respeitando o limite.

### 4.4 — Simulação da política com validação deslizante

**Objetivo:** comparar políticas, não alocações.

Políticas a comparar: calendário anual, calendário trimestral, banda relativa de
25%, banda de 50%, somente aporte sem venda (o comportamento atual), e nunca
rebalancear.

**Método:** validação deslizante. Estimar na janela até `t`, aplicar em `t+1`,
rolar. Nunca usar o conjunto inteiro.

**Reportar por política:** retorno líquido de custos, volatilidade, perda máxima,
número de negócios por ano, custo total pago e imposto total pago.

**Pronto quando:** sai uma tabela política × métrica, e o relatório diz qual
política vence depois de custos — que frequentemente é menos movimento do que a
intuição sugere.

### 4.5 — Custo da transição

Antes de decidir mudar o alvo, calcular o custo de sair da carteira atual para a
proposta: corretagem mais imposto sobre o lucro realizado nas vendas.

Apresentar esse custo como desconto inicial na alternativa. Isso costuma inverter
o resultado de mudanças pequenas e deixar de pé apenas as grandes — que é o
comportamento desejado.

---

## Regras de alocação e referências

Toda regra do `--comparar` e do `--validar` vem da literatura. Onde o Stonks
adapta, a adaptação está dita.

| Regra | Referência | Adaptação no Stonks |
|---|---|---|
| `iguais` (1/N) | DeMiguel, Garlappi e Uppal (2009), *Review of Financial Studies* 22(5) | — |
| `inverso_vol` | forma simples da paridade de risco; ver a linha abaixo | ignora correlações |
| `paridade_risco` | Qian (2005); Maillard, Roncalli e Teïletche (2010), *Journal of Portfolio Management* 36(4) | covariância encolhida (Ledoit-Wolf) |
| `minima_variancia` | Haugen e Baker (1991); Clarke, de Silva e Thorley (2006) | só comprada |
| `markowitz` | Markowitz (1952), *Journal of Finance* 7(1) | máximo Sharpe, só comprada |
| `markowitz_encolhido` | Jorion (1986), *Journal of Financial and Quantitative Analysis* 21(3) | médias por Bayes-Stein |
| `tendencia` | — (formaliza a intuição "o que sobe continua subindo") | inclinação do log-preço em 36 meses |
| `momentum_12_1` | Jegadeesh e Titman (1993), *Journal of Finance* 48(1) | só comprada, peso proporcional ao retorno |

### `momentum_12_1` e a janela da `tendencia` (05/11/2026)

A `tendencia` mede 36 meses, e nesse horizonte a literatura documenta
**reversão**, não continuação: De Bondt e Thaler (1985), *Journal of Finance*
40(3). O momentum documentado usa de 3 a 12 meses. A versão clássica, aqui
implementada, soma o retorno de t-12 a t-2 e pula o último mês, onde há
reversão de curto prazo. A `tendencia` ficou para comparação, não foi apagada.

O artigo original compra o decil vencedor e vende o perdedor entre centenas de
ações. Com 8 ações e só posição comprada, isto é uma adaptação, e o resultado
não testa o artigo. A evidência de momentum fora dos EUA está em Asness,
Moskowitz e Pedersen (2013), *Journal of Finance* 68(3); a evidência
específica para o Brasil **não foi conferida**. Quedas bruscas da estratégia:
Daniel e Moskowitz (2016), *Journal of Financial Economics* 122(2).

**Registrado antes de rodar:** com 8 ativos e cerca de 35 meses avaliados, o
esperado é **empate** das duas regras contra `iguais`. Única aposta: o
`momentum_12_1` gira mais que a `tendencia`, porque a janela curta muda mais
rápido. Se ele vencer com folga, a primeira suspeita é sorte do período, não
descoberta.

---

## Dívida técnica encontrada

Não bloqueia as fases acima, mas vale registrar.

- **`etl.py` está órfão.** Parece superado por `carteira/io_extrato.py`, mas
  continua na raiz e é o único coberto por teste. Decidir: remover, ou deixar
  claro que é ferramenta de inspeção manual.
- **`tests/test_etl.py` depende de `extratos/`**, que está no `.gitignore`. Num
  clone limpo os dois testes falham. Criar um extrato sintético pequeno em
  `tests/dados/` com dados inventados — resolve a dependência sem versionar
  dado pessoal.
- **Cobertura de teste quase nula** justamente nos módulos onde erro é
  silencioso: `retorno.py`, `rebalance.py` e `posicao.py`. O cálculo de TIR e a
  distribuição de troco merecem teste com valores conferidos à mão.
- **`config/` versionado.** A decisão está documentada no `.gitignore`, mas o
  repositório é público e o `alvo.json` expõe sua alocação. Não é dado
  identificável, mas vale uma decisão consciente.

---

## Ordem sugerida

Fase 0 inteira é pré-requisito. Depois 1.1 a 1.4 entregam sozinhas o diagnóstico
que motiva tudo — se o número efetivo de apostas vier perto de 3, a decisão de
diversificar praticamente se resolve ali, sem precisar da Fase 2.

Fase 3 é o que realmente decide. Fase 2 é contexto. Fase 4 só depois que a
alocação estiver definida.

---

*Este documento trata de metodologia quantitativa. Não é recomendação de
investimento nem análise de valores mobiliários.*
