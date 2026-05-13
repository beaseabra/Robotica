# Autonomous Lane Following — RL Robustness Study
### CC3046 — Introduction to Intelligent Robotics
**Beatriz Seabra · Rodrigo Simões · Isabela Britto Cartaxo**

---

## Estrutura do projeto

```
autonomous-lane-following/
├── env/
│   ├── webots_env.py              # Ambiente base Gymnasium (C1)
│   ├── webots_critical_env.py     # Ambiente com obstáculos dinâmicos (C3/C4)
│   ├── lidar_noise_wrapper.py     # Wrapper de ruído LiDAR (C2/C4/C5)
│   ├── discrete_action_wrapper.py # Wrapper de ação discreta (DQN)
│   └── critical_obstacles.py      # Gestor de obstáculos via Supervisor API
├── dqn/
│   ├── train_dqn_c1.py            # Treino DQN — C1 (baseline)
│   ├── train_dqn_c2.py            # Treino DQN — C2 (ruído)
│   ├── train_dqn_c3.py            # Treino DQN — C3 (dinâmicos)
│   └── train_dqn_c4.py            # Treino DQN — C4 (combinado)
├── ppo/
│   ├── train_ppo_c1.py            # Treino PPO — C1 (baseline)
│   ├── train_ppo_c2.py            # Treino PPO — C2 (ruído)
│   ├── train_ppo_c3.py            # Treino PPO — C3 (dinâmicos)
│   └── train_ppo_c4.py            # Treino PPO — C4 (combinado)
├── evaluation/
│   ├── evaluate.py                # Avaliação universal (C1–C4, 100 episódios)
│   └── plot_results.py            # Gera plots e tabela comparativa
├── worlds/
│   ├── city_default.wbt           # Mundo Webots sem obstáculos dinâmicos (C1/C2)
│   └── city_obstacles.wbt         # Mundo Webots com PEDESTRIAN_1 e VEHICLE_1 (C3/C4)
├── models/                        # Modelos guardados (.zip) — criado automaticamente
├── logs/                          # TensorBoard logs — criado automaticamente
├── results/                       # JSONs + plots — criado automaticamente
└── requirements.txt
```

---

## 1. Instalação

### 1.1 Instalar Webots

Descarregar Webots R2023b ou superior em: https://cyberbotics.com/

### 1.2 Instalar dependências Python

```bash
cd autonomous-lane-following

# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows (PowerShell)

# Instalar pacotes
pip install -r requirements.txt
```

### 1.3 Definir a variável WEBOTS_HOME

O código usa esta variável para localizar a biblioteca Python do Webots.

```bash
# Linux
export WEBOTS_HOME="/usr/local/webots"

# macOS
export WEBOTS_HOME="/Applications/Webots.app/Contents"

# Windows (PowerShell)
$env:WEBOTS_HOME = "C:\Program Files\Webots"
```

Para não ter de definir sempre, adicionar ao `.bashrc` / `.zshrc` / perfil PowerShell.

---

## 2. Como ligar ao Webots

O ambiente Gymnasium **não lança o Webots automaticamente**. O processo é sempre:

1. **Abrir o Webots** manualmente
2. **Abrir o mundo correto** (File → Open World)
3. **Correr o script Python** — o Webots fica à espera da ligação do controller

O script Python liga-se ao Webots via `Supervisor()` quando é instanciado.

### Qual mundo abrir para cada condição

| Condição | Mundo a abrir no Webots |
|----------|------------------------|
| C1 — baseline | `worlds/city_default.wbt` |
| C2 — ruído | `worlds/city_default.wbt` |
| C3 — dinâmicos | `worlds/city_obstacles.wbt` |
| C4 — combinado | `worlds/city_obstacles.wbt` |
| C5 (eval C4) | `worlds/city_obstacles.wbt` |

> **Nota:** `city_obstacles.wbt` contém os nós `PEDESTRIAN_1` e `VEHICLE_1`
> necessários para C3/C4. Se abrir `city_default.wbt` com C3/C4,
> o `CriticalObstacleManager` imprime avisos e ignora os obstáculos
> (sem crash — mas os cenários não funcionam).

### Controller do Webots

O BmwX5 no mundo deve ter o controller configurado como `<extern>` ou
`extern` para que o Python externo tome o controlo. Verificar no painel
de propriedades do nó no Webots.

---

## 3. Condições experimentais

| Cond. | Sensores | Obstáculos | Stack de ambiente |
|-------|----------|------------|-------------------|
| C1 | Limpos | Estáticos | `WebotsVehicleEnv()` |
| C2 | Ruído Gaussiano + dropout | Estáticos | `LiDARNoiseWrapper(WebotsVehicleEnv())` |
| C3 | Limpos | Dinâmicos | `WebotsCriticalEnv()` |
| C4 | Ruído Gaussiano + dropout | Dinâmicos | `LiDARNoiseWrapper(WebotsCriticalEnv())` |
| C5 | Treino em C2, teste em C4 | Dinâmicos | Modelo C2, stack C4 |

**Parâmetros de ruído:** `noise_std=0.1`, `dropout_prob=0.05`

---

## 4. Treino

### Ordem recomendada para o checkpoint

Começar por C1 (sem obstáculos, sem ruído). Depois C2 se houver tempo.
C3 e C4 ficam para após o checkpoint.

**Passo a passo:**

#### C1 — Baseline (abrir city_default.wbt)

```bash
# Terminal 1: abrir Webots com city_default.wbt

# Terminal 2:
python dqn/train_dqn_c1.py    # DQN — ~2 a 4 horas (800k steps)
# Quando terminar:
python ppo/train_ppo_c1.py    # PPO — ~2 a 4 horas (800k steps)
```

Modelos guardados em:
- `models/dqn_c1/dqn_c1_final.zip`
- `models/ppo_c1/ppo_c1_final.zip`

#### C2 — Sensor Noise (abrir city_default.wbt)

```bash
python dqn/train_dqn_c2.py
python ppo/train_ppo_c2.py
```

#### C3 — Dynamic Obstacles (abrir city_obstacles.wbt)

```bash
python dqn/train_dqn_c3.py
python ppo/train_ppo_c3.py
```

#### C4 — Combined (abrir city_obstacles.wbt)

```bash
python dqn/train_dqn_c4.py
python ppo/train_ppo_c4.py
```

---

## 5. Avaliação (100 episódios por condição)

O script `evaluation/evaluate.py` é universal — suporta todas as condições e algoritmos.

```bash
# ── C1 ────────────────────────────────────────────────────────────
python evaluation/evaluate.py \
  --algo dqn --condition c1 \
  --model ./models/dqn_c1/dqn_c1_final \
  --n_episodes 100

python evaluation/evaluate.py \
  --algo ppo --condition c1 \
  --model ./models/ppo_c1/ppo_c1_final \
  --n_episodes 100

# ── C2 (abrir city_default.wbt) ───────────────────────────────────
python evaluation/evaluate.py \
  --algo dqn --condition c2 \
  --model ./models/dqn_c2/dqn_c2_final \
  --n_episodes 100

python evaluation/evaluate.py \
  --algo ppo --condition c2 \
  --model ./models/ppo_c2/ppo_c2_final \
  --n_episodes 100

# ── C3 (abrir city_obstacles.wbt) ─────────────────────────────────
python evaluation/evaluate.py \
  --algo dqn --condition c3 \
  --model ./models/dqn_c3/dqn_c3_final \
  --n_episodes 100

python evaluation/evaluate.py \
  --algo ppo --condition c3 \
  --model ./models/ppo_c3/ppo_c3_final \
  --n_episodes 100

# ── C4 (abrir city_obstacles.wbt) ─────────────────────────────────
python evaluation/evaluate.py \
  --algo dqn --condition c4 \
  --model ./models/dqn_c4/dqn_c4_final \
  --n_episodes 100

python evaluation/evaluate.py \
  --algo ppo --condition c4 \
  --model ./models/ppo_c4/ppo_c4_final \
  --n_episodes 100

# ── C5 — PPO treinado em C2, avaliado em C4 ───────────────────────
# (abrir city_obstacles.wbt)
python evaluation/evaluate.py \
  --algo ppo --condition c4 \
  --model ./models/ppo_c2/ppo_c2_final \
  --n_episodes 100 --tag c5
```

Resultados guardados em `results/` como JSON (ex: `results/dqn_c1.json`).

### Avaliar modelo C1 nas condições C2/C3/C4 (cross-condition)

Para medir a degradação de performance sem retreino:

```bash
# Modelo C1 treinado, avaliado em C2 (zero-shot noise robustness)
python evaluation/evaluate.py \
  --algo dqn --condition c2 \
  --model ./models/dqn_c1/dqn_c1_final \
  --n_episodes 50 --tag cross_c1_in_c2
```

---

## 6. Plots e tabela

```bash
python evaluation/plot_results.py
```

Gera na pasta `results/plots/`:
- `success_rate.png`   — barchart taxa de sucesso por condição e algoritmo
- `lane_deviation.png` — boxplot desvio lateral
- `avg_steps.png`      — duração média dos episódios
- `results/summary_table.csv` — tabela completa de métricas

---

## 7. TensorBoard (learning curves)

```bash
tensorboard --logdir ./logs/
# Abrir http://localhost:6006
```

---

## 8. Métricas recolhidas por episódio

| Métrica | Descrição |
|---------|-----------|
| `success` | Episódio completado sem colisão |
| `collision` | Colisão detectada (LiDAR min < 0.45 m) |
| `steps` | Passos no episódio antes de terminar |
| `total_reward` | Reward acumulado |
| `lane_deviation` | Erro lateral médio normalizado (0–1) |
| `episode_time_s` | Tempo real do episódio (segundos) |

**Nota:** A colisão é detetada via `info["collision"]` propagado pelo ambiente
(LiDAR min < 0.45 m). Nunca se usa o valor do reward para inferir colisões.

---

## 9. Sementes (reprodutibilidade)

Todos os scripts usam **seed = 42**:
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `env.reset(seed=42 + episode_number)` na avaliação

---

## 10. Parâmetros dos algoritmos

### DQN
| Parâmetro | Valor |
|-----------|-------|
| `learning_rate` | 1e-4 |
| `buffer_size` | 100 000 |
| `learning_starts` | 5 000 |
| `batch_size` | 64 |
| `gamma` | 0.99 |
| `exploration_fraction` | 0.2 |
| `exploration_final_eps` | 0.05 |
| `net_arch` | [256, 256] |
| `total_timesteps` | 800 000 |

### PPO
| Parâmetro | Valor |
|-----------|-------|
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `net_arch` | [256, 256] |
| `total_timesteps` | 800 000 |

### Ruído LiDAR (C2 / C4 / C5)
| Parâmetro | Valor |
|-----------|-------|
| `noise_std` | 0.1 |
| `dropout_prob` | 0.05 (5% dos raios por timestep) |

---

## 11. Notas técnicas

**Bug fix (critical_obstacles.py):** O cálculo de distância usa os índices `[0, 2]`
(plano X-Z), que é o plano do chão no Webots (Y é o eixo vertical). Uma versão
anterior usava `[0, 1]` erradamente.

**Checkpoints automáticos:** Os scripts de treino guardam checkpoints a cada
10 000 steps via `CheckpointCallback`. Em caso de crash do Webots, pode-se
continuar a partir do último checkpoint (ver `models/<algo>_<cond>/`).

**check_env:** Todos os scripts de treino correm `check_env(env)` antes de treinar.
Se falhar, o ambiente tem um problema — resolver antes de avançar.

**Sem GPU?** O treino funciona em CPU mas demora mais. Para cada condição
contar com 3–6 horas. C4 pode demorar mais.
