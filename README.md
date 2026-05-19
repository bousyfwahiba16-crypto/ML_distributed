# Projet Systèmes Répartis — Entraînement Distribué de Modèles d'Apprentissage Profond

**TP — Guide de reproduction**

Master d'Excellence en Intelligence Artificielle
Faculté des Sciences Ben M'Sik — Université Hassan II de Casablanca

---

## Sommaire

1. [Présentation](#1-présentation)
2. [Vue d'ensemble des parcours](#2-vue-densemble-des-parcours)
3. [Prérequis](#3-prérequis)
4. [Structure du dépôt](#4-structure-du-dépôt)
5. [Installation commune](#5-installation-commune)
6. [Parcours A — Étudiants sans GPU](#6-parcours-a--étudiants-sans-gpu)
   - 6.1 [Validation rapide sur Google Colab (CPU)](#61-validation-rapide-sur-google-colab-cpu)
   - 6.2 [Exécution locale CPU (Windows ou Linux)](#62-exécution-locale-cpu-windows-ou-linux)
7. [Parcours B — Étudiants avec GPU](#7-parcours-b--étudiants-avec-gpu)
   - 7.1 [Validation rapide sur Google Colab (GPU)](#71-validation-rapide-sur-google-colab-gpu)
   - 7.2 [Exécution locale GPU (WSL Ubuntu requis pour le distribué)](#72-exécution-locale-gpu-wsl-ubuntu-requis-pour-le-distribué)
8. [Résultats attendus et livrables](#8-résultats-attendus-et-livrables)
9. [Dépannage (FAQ)](#9-dépannage-faq)
10. [Auteurs](#10-auteurs)

---

## 1. Présentation

Ce TP met en œuvre un pipeline d'**entraînement distribué** d'un modèle d'apprentissage profond, en exploitant les primitives de communication collective de PyTorch (`torch.distributed`) via l'API `DistributedDataParallel` (DDP).

Deux backends sont utilisés selon le matériel :

| Backend | Cible | Plateformes supportées |
|---------|-------|------------------------|
| **`gloo`** | CPU | Windows &nbsp;&nbsp;&nbsp; Linux &nbsp;&nbsp;&nbsp; macOS |
| **`nccl`** | GPU NVIDIA | Linux &nbsp;&nbsp;&nbsp; Windows → **WSL requis** |

L'objectif pédagogique est de **comparer expérimentalement** :

- l'entraînement **mono-processus** (référence),
- l'entraînement **distribué multi-processus** (DDP),

et d'analyser l'efficacité du scaling (speedup, throughput, temps par epoch) pour chacun des deux backends.

---

## 2. Vue d'ensemble des parcours

| Parcours | Matériel | Validation rapide | Exécution complète | Entraînement distribué |
|----------|----------|-------------------|--------------------|------------------------|
| **A — Sans GPU** | CPU uniquement | `notebooks/SysRepartisCPU.ipynb` sur Colab | `scripts/CPU/*.py` en local | via backend `gloo` (fonctionne sous **Windows ET Linux** sans WSL) |
| **B — Avec GPU** | GPU NVIDIA | `notebooks/SysRepartisGPU.ipynb` sur Colab | `scripts/GPU/*.py` en local | via backend `nccl` — **WSL Ubuntu obligatoire** sous Windows |

> **Important — différence Windows entre CPU et GPU distribués.**
> Le backend `gloo` (utilisé pour le CPU distribué) **fonctionne nativement sous Windows**. Les étudiants sans GPU peuvent donc faire de l'entraînement distribué directement sous Windows, **sans WSL**.
> En revanche, le backend `nccl` (requis pour le GPU distribué) **n'est pas disponible sous Windows natif**. Les étudiants avec GPU doivent passer par **WSL 2 Ubuntu** pour la partie distribuée.

> **Limite Google Colab.** Quel que soit le notebook utilisé (CPU ou GPU), Colab **ne permet pas l'entraînement distribué multi-processus** : une seule instance est allouée par session et les ports requis par `torch.distributed` sont bloqués. Les notebooks servent donc à **valider la chaîne complète** (chargement, modèle, mono-processus, évaluation) avant de lancer le distribué en local.

---

## 3. Prérequis

### 3.1 Prérequis communs

- **Python** ≥ 3.10
- Compte **Google** (pour Colab)

### 3.2 Parcours B — Avec GPU

- **GPU NVIDIA** compatible CUDA
- **WSL 2** activé sous Windows 10 (build ≥ 19044) ou Windows 11
- Distribution **Ubuntu 22.04** installée dans WSL
- **CUDA Toolkit** ≥ 12.1 installé dans WSL

---

## 4. Structure du dépôt

```
sysRepartis/
├── notebooks/                      # Parcours A — Notebooks Colab
│   ├── SysRepartisCPU.ipynb        
│   └── SysRepartisGPU.ipynb
│
├── outputs/                        # Généré à l'exécution
|
├── scripts/                        # Scripts de lancement
│   ├── CPU
│   │   ├── run_distributed.py              # Chargement / prétraitement
│   │   └── run_single.py   
│   └── GPU
│         ├── run_distri_GPU.py              # Chargement / prétraitement
│         └── run_single_GPU.py   
│
├── src/                            # Parcours B — Scripts Python
│   ├── data/
│   │   ├── data_utils.py              # Chargement / prétraitement
│   │   └── MNIST/        
│   ├── models/
│   │   └── model.py                # Définition du modèle
│   └── training/
│        ├── train.py         # Entraînement mono-GPU/CPU
│        └── training_Gpu.py    # Entraînement distribué (DDP)
│
├── requirements.txt                # Dépendances Python
└── README.md                       

```

---

## 5. Installation commune

### 5.1 Cloner le dépôt

```bash
git clone https://github.com/bousyfwahiba16-crypto/ML_distributed.git
cd ML_distributed
```

### 5.2 Créer un environnement virtuel

**Sous Linux / macOS / WSL :**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Sous Windows (PowerShell) :**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 5.3 Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.4 Vérifier l'installation

**Test CPU (parcours A) :**

```bash
python -c "import torch; import torch.distributed as dist; print('PyTorch :', torch.__version__); print('Backend gloo disponible :', dist.is_gloo_available())"
```

Sortie attendue :
```
PyTorch : 2.x.x
Backend gloo disponible : True
```

**Test GPU (parcours B) :**

```bash
python -c "import torch; import torch.distributed as dist; print('CUDA disponible :', torch.cuda.is_available()); print('Nombre de GPU :', torch.cuda.device_count()); print('Backend nccl disponible :', dist.is_nccl_available())"
```

Sortie attendue (sous WSL Ubuntu) :
```
CUDA disponible : True
Nombre de GPU : >= 1
Backend nccl disponible : True
```

---

## 6. Parcours A — Étudiants sans GPU

### 6.1 Validation rapide sur Google Colab (CPU)

> Objectif : valider la chaîne complète sans installation locale. L'entraînement distribué n'est pas exécutable sur Colab.

#### 6.1.1 Ouvrir le notebook sur Colab

1. Se rendre sur [https://colab.research.google.com](https://colab.research.google.com).
2. **Fichier → Ouvrir un notebook → onglet GitHub**.
3. Coller l'URL du dépôt et sélectionner `notebooks/SysRepartisCPU.ipynb`.

#### 6.1.2 Configurer l'environnement d'exécution

`Exécution → Modifier le type d'exécution → Accélérateur matériel : CPU`.

#### 6.1.3 Cellule d'initialisation

Le notebook commence par :

```python
!git clone https://github.com/<utilisateur>/sysRepartis.git
%cd sysRepartis
!pip install -q -r requirements.txt
```

Exécuter ensuite toutes les cellules dans l'ordre.

### 6.2 Exécution locale CPU (Windows ou Linux)

#### 6.2.1 Entraînement mono-processus (référence)

```bash
python scripts/CPU/run_single.py
```
OR

```bash
python -m scripts.CPU.run_single
```

#### 6.2.2 Entraînement distribué CPU (backend `gloo`)

Le backend `gloo` fonctionne nativement sous Windows ET Linux, sans WSL.

**Lancement via `torchrun` :**

```bash
torchrun --standalone --nproc_per_node=<2> scripts/CPU/run_distributed.py
```
OR

```bash
python -m scripts.CPU.run_distributed
```
**Exemple avec 2 processus parallèles (2 Cores)**


> Sur CPU, le nombre de processus pertinent dépend du nombre de cœurs physiques. Une valeur entre 2 et `cœurs_physiques / 2` donne généralement les meilleurs résultats (au-delà, le surcoût de communication annule le gain).

#### 6.2.3 Spécificité Windows

Sous PowerShell, la commande est identique :

```powershell
torchrun --standalone --nproc_per_node=4 scripts\CPU\run_distributed.py
```

Aucune configuration supplémentaire n'est requise : le backend `gloo` est inclus dans PyTorch dès l'installation via `pip install torch`.

---

## 7. Parcours B — Étudiants avec GPU

### 7.1 Validation rapide sur Google Colab (GPU)

> Objectif : **valider la chaîne GPU** sur un GPU Colab (T4). L'entraînement distribué GPU n'est pas exécutable sur Colab.

#### 7.1.1 Ouvrir le notebook

1. [https://colab.research.google.com](https://colab.research.google.com) → **Fichier → Ouvrir un notebook → onglet GitHub**.
2. Sélectionner `notebooks/SysRepartisGPU.ipynb`.

#### 7.1.2 Activer le GPU

`Exécution → Modifier le type d'exécution → Accélérateur matériel : T4 GPU`.

#### 7.1.3 Cellule d'initialisation

```python
!git clone https://github.com/<utilisateur>/sysRepartis.git
%cd sysRepartis
!pip install -q -r requirements.txt
!nvidia-smi
```

### 7.2 Exécution locale GPU (WSL Ubuntu requis pour le distribué)

> **Pourquoi WSL ?** PyTorch utilise **NCCL** comme backend de communication collective pour DDP sur GPU. NCCL n'est disponible que sous Linux. Sous Windows natif, l'entraînement **distribué GPU est impossible**. L'entraînement **mono-GPU** fonctionne en revanche sous Windows natif.

#### 7.2.1 Préparer WSL 2 Ubuntu

Dans une console **PowerShell exécutée en administrateur** :

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Redémarrer si demandé, puis ouvrir **Ubuntu** depuis le menu Démarrer pour finaliser la création du compte.

#### 7.2.2 Installer les outils sous Ubuntu

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip git build-essential
```

#### 7.2.3 Vérifier l'accès au GPU depuis WSL

```bash
nvidia-smi
```

Si la commande échoue :

- Mettre à jour le **pilote NVIDIA Windows** (≥ 535) depuis [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx).
- Installer le **CUDA Toolkit pour WSL** :

  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-4
  ```

- Ajouter CUDA au `PATH` :

  ```bash
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc
  ```

#### 7.2.4 Récupérer le projet sous WSL

```bash
cd ~
git clone https://github.com/<utilisateur>/sysRepartis.git
cd sysRepartis
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 7.2.5 Entraînement mono-GPU (référence)

```bash
python scripts/GPU/run_single_GPU.py
```

#### 7.2.6 Entraînement distribué GPU (backend `nccl`)

**Doit obligatoirement être exécuté sous WSL Ubuntu ou Linux natif.**

```bash
torchrun --standalone --nproc_per_node=<NB_GPU> scripts/GPU/run_distri_GPU.py
```

**Exemple avec 2 GPU :**

```bash
torchrun --standalone --nproc_per_node=2 scripts/GPU/run_distri_GPU.py
```

---

## 8. Résultats attendus et livrables

À l'issue d'une exécution complète, les sorties suivantes sont produites :

## Configuration matérielle

| Composant | Spécification |
|-----------|---------------|
| CPU | AMD Ryzen 5 5600G with Radeon Graphics (6 cœurs / 12 threads, 3,90 GHz) |
| GPU | NVIDIA GeForce RTX 4060 Ti (8 Go VRAM) |
| Mémoire | 16 Go |
| OS | Windows (exécution via PowerShell) |

---

## 1. Entraînement CPU mono-processus (référence)

**Commande :**
```powershell
python .\run_single.py
```

**Sortie console :**
```
Epoch 0 | Loss: 0.1956
Temps d'entraînement (single): 34.0s
```

**Observation système :** utilisation CPU ~3 % avant exécution, montant légèrement pendant le run.

---

## 2. Entraînement CPU distribué (`world_size = 2`)

**Commande :**
```powershell
python .\run_distributed.py
```

**Sortie console :**
```
[Rank 0] Epoch 0 | Loss: 0.1837
[Rank 0] Epoch 1 | Loss: 0.0470
Temps d'entraînement (world_size=2): 60.8s
```

**Observation système :** pic CPU à **81 %** pendant l'exécution (vs 3 % au repos), confirmant la parallélisation effective des deux processus.

---

## 3. Entraînement GPU mono-processus

**Commande :**
```powershell
python .\run_single_gpu.py
```

**Sortie console :**
```
True
NVIDIA GeForce RTX 4060 Ti
NCCL: False
Utilisation de: cuda
Epoch 0 | Loss: 0.1097
Epoch 1 | Loss: 0.0479
Temps d'entraînement GPU (single): 20.4s
```

**Observation système :** GPU NVIDIA RTX 4060 Ti détecté et actif (`cuda`). Le drapeau `NCCL: False` confirme que **NCCL n'est pas disponible sous Windows natif**, ce qui justifie l'usage de WSL Ubuntu pour la distribution GPU.

---

## Synthèse comparative

| Mode | Backend | Epochs | Loss finale | Temps total | Speedup vs CPU single |
|------|---------|--------|-------------|-------------|------------------------|
| CPU — mono-processus | `gloo` (1 proc) | 1 | 0,1956 | **34,0 s** | 1,00× (référence) |
| CPU — distribué `world_size=2` | `gloo` | 2 | 0,0470 | 60,8 s | 0,56× |
| GPU — mono-processus | CUDA | 2 | 0,0479 | **20,4 s** | 1,67× |

---

## Interprétations

- **CPU distribué plus lent que CPU single** : avec un modèle léger (MNIST) et seulement 6 cœurs physiques partagés entre 2 processus + dataloaders, le surcoût de synchronisation `all-reduce` via `gloo` dépasse le gain de parallélisation. Ce résultat est attendu et **illustre la limite du speedup naïf** : la distribution n'a d'intérêt qu'au-delà d'un seuil de charge calculatoire.

- **GPU single ≈ 1,7× plus rapide que CPU single** malgré 2 epochs (vs 1 epoch côté CPU single), soit un facteur effectif d'environ **3,3× par epoch**. La loss finale (0,0479) est cohérente avec le CPU distribué (0,0470), validant l'équivalence numérique des trois pipelines.

- **`NCCL: False` sous Windows** : confirme empiriquement la nécessité de basculer sous **WSL Ubuntu** pour exécuter l'entraînement distribué GPU (DDP + NCCL), comme indiqué dans le README principal (§7.1).

---

## Captures d'écran de référence

Les exécutions ci-dessus sont documentées par captures du terminal PowerShell et du Gestionnaire des tâches Windows (onglet Performance) montrant l'utilisation CPU/GPU pendant chaque run. Ces captures sont disponibles dans le repertoire outputs/ avec un video de demonstration.

---

## 9. Dépannage (FAQ)

| Problème | Cause probable | Solution |
|----------|----------------|----------|
| `RuntimeError: Distributed package doesn't have NCCL built in` | Lancement de `run_distri_GPU.py` sous Windows natif | Passer obligatoirement par WSL Ubuntu (cf. §7.2) |
| `gloo` lent par rapport au mono-processus CPU | Trop de processus pour le nombre de cœurs, ou batch trop petit | Réduire `--nproc_per_node` à `cœurs_physiques / 2` ; augmenter `batch_size` |
| `torch.cuda.is_available()` retourne `False` sous WSL | Pilote NVIDIA Windows obsolète ou CUDA Toolkit absent | Mettre à jour le pilote Windows ≥ 535 et installer `cuda-toolkit-12-4` |
| `CUDA out of memory` | `batch_size` trop élevé pour la VRAM | Réduire `batch_size` ; activer `gradient_accumulation_steps` |
| `Address already in use` (port 29500) | Précédent processus `torchrun` encore actif | `pkill -f torchrun` (Linux/WSL) ou `taskkill /F /IM python.exe` (Windows) ; ou changer `--rdzv_endpoint=localhost:29501` |
| Processus DDP figés au démarrage (multi-nœuds) | Pare-feu, IP du maître inaccessible | Tester avec `ping <IP_MAITRE>` puis `telnet <IP_MAITRE> 29500` |
| Notebooks Colab très lents | Mauvais type d'exécution sélectionné | Vérifier `Exécution → Type d'exécution` (CPU pour `SysRepartisCPU`, GPU T4 pour `SysRepartisGPU`) |
| `ImportError: libcuda.so.1` (WSL) | `LD_LIBRARY_PATH` non configuré | Ré-exécuter les exports du §7.2.3 puis `source ~/.bashrc` |
| Sous Windows, `torchrun` introuvable | `Scripts/` du venv non dans le `PATH` | Activer le venv (`.venv\Scripts\Activate.ps1`) avant de lancer |
| Divergence des résultats entre runs distribués | Graine aléatoire non fixée par rang | Utiliser `torch.manual_seed(seed + rank)` et `DistributedSampler(seed=...)` |

---

## 10. Auteurs

**Équipe projet :**
- Wahiba Bousyf et Hajar Alouani

Master d'Excellence en Intelligence Artificielle
Faculté des Sciences Ben M'Sik — Université Hassan II de Casablanca
Année universitaire 2025–2026
