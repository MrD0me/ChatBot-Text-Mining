# ChatBot Text Mining — ATLAS

**ATLAS** è un chatbot che risponde in italiano alle domande sui tuoi PDF. Carichi uno o più
documenti, fai una domanda e ottieni una risposta basata solo sul loro contenuto. Ogni risposta
cita il file e la pagina da cui proviene e mostra l'anteprima della pagina con il passaggio
evidenziato. Se nei documenti l'informazione non c'è, il chatbot lo dice invece di inventare.

Usa la tecnica **RAG** (Retrieval-Augmented Generation):

1. il testo dei PDF viene diviso in passaggi e indicizzato **sul tuo computer**, con il modello
   di embedding BGE-M3 e con TF-IDF (ricerca ibrida);
2. per ogni domanda vengono cercati i passaggi più pertinenti;
3. i passaggi trovati vengono inviati a un modello linguistico tramite **Groq** (API gratuita),
   che scrive la risposta citando le fonti.

L'interfaccia è un'app web locale fatta con Streamlit.

| Cartella | Contenuto |
|---|---|
| [`ChatBot_semplice/`](ChatBot_semplice/) | Il chatbot: app, codice RAG, notebook, test e relazione |

Dettagli tecnici: [README del chatbot](ChatBot_semplice/README.md) e
[relazione](ChatBot_semplice/RELAZIONE.md).

---

## Installazione passo passo (Windows)

### 1. Requisiti

- **Python 3.13** (è la versione con cui è stato testato), da [python.org](https://www.python.org/downloads/).
  Durante l'installazione spunta **"Add python.exe to PATH"**.
- **Git**, da [git-scm.com](https://git-scm.com/downloads). In alternativa puoi scaricare il
  progetto come zip (vedi il passo 2).
- Circa **5 GB liberi** su disco: le librerie e il modello BGE-M3 (circa 2,3 GB), che viene
  scaricato al primo avvio.
- Una connessione a Internet, per il primo avvio e per le risposte di Groq.
- Facoltativa: una **GPU NVIDIA**, che rende molto più veloce l'indicizzazione dei PDF. Senza GPU
  funziona lo stesso, usando il processore.

Per controllare che Python sia installato, apri **PowerShell** e scrivi:

```powershell
python --version
```

Deve comparire `Python 3.13.x`.

### 2. Scarica il progetto

Con Git, dalla cartella dove vuoi salvarlo (per esempio `Documenti`):

```powershell
git clone https://github.com/MrD0me/ChatBot-Text-Mining.git
```

Senza Git: sulla pagina GitHub del progetto premi **Code → Download ZIP** ed estrai lo zip.

Poi entra nella cartella del chatbot:

```powershell
cd ChatBot-Text-Mining\ChatBot_semplice
```

Tutti i comandi che seguono vanno lanciati **da questa cartella**.

### 3. Crea l'ambiente Python

L'ambiente virtuale `.venv` tiene le librerie del progetto separate dal resto del computer:

```powershell
python -m venv .venv
```

Se hai più versioni di Python installate, usa `py -3.13 -m venv .venv` al posto del comando sopra.

### 4. Installa le librerie

**Solo se hai una GPU NVIDIA**, installa prima PyTorch con il supporto CUDA:

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Poi, **in ogni caso**, installa le librerie del progetto:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

L'installazione può richiedere alcuni minuti.

### 5. Ottieni la chiave API di Groq (gratuita)

1. Vai su [console.groq.com/keys](https://console.groq.com/keys) e accedi (anche con Google o GitHub).
2. Premi **Create API Key** e copia la chiave, che inizia con `gsk_`.

### 6. Inserisci la chiave nel file `.env`

Crea il file `.env` partendo dall'esempio e aprilo nel Blocco note:

```powershell
Copy-Item .env.example .env
```

```powershell
notepad .env
```

Incolla la chiave dopo l'uguale e salva:

```
GROQ_API_KEY=gsk_la_tua_chiave
```

Il file `.env` resta solo sul tuo computer: è escluso da Git e **non va mai condiviso**.
In alternativa puoi saltare questo passo e incollare la chiave direttamente nell'app, nel
selettore del modello sotto la casella delle domande; in quel caso vale solo finché l'app resta aperta.

### 7. Avvia il chatbot

```powershell
.venv\Scripts\streamlit run app.py
```

Il browser si apre da solo su **http://localhost:8501**. Se non si apre, copia l'indirizzo nel browser.

Al **primo avvio** viene scaricato il modello BGE-M3 (circa 2,3 GB): aspetta che finisca. Dalle
volte successive l'avvio è rapido.

### 8. Usa il chatbot

1. Nella barra laterale premi **Scegli i file** (o trascina i PDF) e carica uno o più documenti.
2. Aspetta che vengano indicizzati. Gli embedding vengono salvati nella cartella `.cache/`,
   così ricaricare lo stesso PDF è immediato.
3. Scrivi la domanda nella casella in basso e premi Invio.
4. Sotto la risposta trovi le fonti numerate: aprine una per vedere la pagina originale con il
   passaggio evidenziato.

### 9. Chiudi e riavvia

Per chiudere il chatbot, torna su PowerShell e premi **Ctrl+C**.

Le volte successive bastano due comandi, dalla cartella in cui hai scaricato il progetto:

```powershell
cd ChatBot-Text-Mining\ChatBot_semplice
```

```powershell
.venv\Scripts\streamlit run app.py
```

---

## macOS e Linux

I passaggi sono gli stessi; cambiano solo alcuni comandi:

| Passo | Comando |
|---|---|
| Crea l'ambiente | `python3 -m venv .venv` |
| Installa le librerie | `.venv/bin/python -m pip install -r requirements.txt` |
| Crea il file `.env` | `cp .env.example .env` |
| Avvia | `.venv/bin/streamlit run app.py` |

Su macOS il passo di PyTorch con CUDA non serve.

---

## Problemi frequenti

| Problema | Soluzione |
|---|---|
| `python` non è riconosciuto | Python non è nel PATH: reinstallalo spuntando "Add python.exe to PATH", oppure usa `py` al posto di `python`. |
| `.venv\Scripts\streamlit` non trovato | Non sei nella cartella `ChatBot_semplice`, oppure i passi 3 e 4 non sono stati completati. |
| L'app chiede la chiave Groq | Il file `.env` manca o la chiave è scritta male: controlla il passo 6. |
| Errore di limite richieste di Groq | Il piano gratuito ha limiti al minuto e al giorno, separati per modello: scegli un altro modello dal menu o riprova più tardi. |
| Il PDF non dà risultati | I PDF scansionati (immagini senza testo) non sono supportati, perché servirebbe l'OCR. L'app li segnala. |
| Indicizzazione molto lenta | Senza GPU BGE-M3 gira sul processore: con PDF lunghi servono alcuni minuti. |

## Test

```powershell
.venv\Scripts\python -m pytest -q
```

Sono 30 test che non richiedono né il modello né la chiave API.
