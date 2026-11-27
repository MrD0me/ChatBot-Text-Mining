# ATLAS (versione semplificata) — chatbot RAG sui tuoi PDF

Fai domande in italiano su uno o più PDF. Ogni risposta indica il file e la pagina da cui
proviene, con l'anteprima della pagina e il passaggio evidenziato.

Rispetto alla versione completa (`..\ChatBot_completo`) questa versione:
- usa **un solo servizio, Groq**, invece di Gemini e Groq intercambiabili;
- non ha la **memoria della conversazione**: ogni domanda è indipendente dalle precedenti,
  come nel notebook di partenza.

```
PDF ─► pagine ─► chunk (450 token, overlap 75) ─► indice ibrido (BGE-M3 + TF-IDF)
                                                        │
domanda ─────────────────────► top-k passaggi (RRF) ─► LLM Groq
                                                        │
                       risposta JSON {risposta, fonti} ─► verifica delle fonti ─► [1] file, pagina
```

## Installazione (Windows)

L'ambiente Python `.venv` sta nella cartella superiore ed è condiviso con la versione completa.
Se non c'è ancora, dalla cartella `ChatBot Text Mining`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r ChatBot_semplice\requirements.txt
```

La riga con `torch` serve solo se hai una GPU NVIDIA. Senza GPU puoi saltarla, e BGE-M3 gira su CPU.

## Chiave API (gratuita)

1. Crea una chiave su [Groq](https://console.groq.com/keys).
2. Aprila nel file `.env` di questa cartella e incollala dopo l'uguale:
   ```
   GROQ_API_KEY=gsk_...
   ```
In alternativa puoi incollarla nella barra laterale dell'app: vale solo per la sessione.
Il file `.env` non va mai condiviso né caricato online.

## Avvio

Va lanciata **da questa cartella**, perché Streamlit legge il tema da `.streamlit/` nella
cartella corrente:

```powershell
cd "C:\Users\dome-\Documents\ChatBot Text Mining\ChatBot_semplice"
..\.venv\Scripts\streamlit run app.py
```

Si apre `http://localhost:8501`. Al primo avvio viene scaricato BGE-M3 (circa 2,3 GB).
Gli embedding dei PDF vengono salvati in `.cache/`, così ricaricare lo stesso file è immediato.

Opzionale: con la variabile `RAG_CARTELLA_PDF=<cartella>` i PDF di quella cartella vengono
caricati all'avvio, comodo per una demo.

## Struttura

| File | Ruolo |
|---|---|
| `app.py` | Interfaccia Streamlit |
| `rag/pdf.py` | Estrazione del testo per pagina, anteprima con evidenziazione |
| `rag/index.py` | Chunking, embedding (con cache), TF-IDF, ricerca ibrida RRF |
| `rag/llm.py` | Chiamata a Groq con output JSON vincolato e gestione degli errori |
| `rag/chat.py` | Prompt, verifica e numerazione delle citazioni |
| `rag/config.py` | Tutti i parametri (chunk, top-k, pesi RRF, modelli Groq) |
| `notebook/ChatBot_RAG.ipynb` | La stessa pipeline passo per passo (locale o Colab) |
| `eval.py` | Valutazione su `TextMining_2026-27_DayOne.pdf` |
| `tests/test_rag.py` | 30 test offline (nessun modello né API) |
| `RELAZIONE.md` | Cosa è stato fatto, perché, e le differenze dal notebook di partenza |

## Test e valutazione

```powershell
..\.venv\Scripts\python -m pytest -q
..\.venv\Scripts\python eval.py percorso\TextMining_2026-27_DayOne.pdf
..\.venv\Scripts\python eval.py percorso\TextMining_2026-27_DayOne.pdf --risposte
```

## Limiti

- I PDF scansionati (pagine senza testo) richiedono OCR, che non è gestito: l'app li segnala.
- Il piano gratuito di Groq ha limiti di richieste al minuto e al giorno. Se si esauriscono,
  si può scegliere un altro modello dal menu, perché i limiti sono per modello.
- Le domande di seguito ("e la seconda?") non funzionano: bisogna scrivere ogni domanda
  per intero.
