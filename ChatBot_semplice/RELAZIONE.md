# Relazione di progetto — Chatbot RAG su documenti PDF (versione semplificata)

*Corso di Text Mining 2026-27*

> **Stato del documento.** La relazione viene aggiornata man mano che il progetto avanza.
> Le sezioni contrassegnate con *(da completare)* attendono dati che non sono ancora
> disponibili: le prove con la chiave Groq e la valutazione sul PDF del corso.

---

## 1. Obiettivo

Realizzare un chatbot che, dati **uno o più documenti PDF**, risponda in italiano a domande
sul loro contenuto tramite **Retrieval-Augmented Generation (RAG)**. Ogni risposta deve
indicare il file e la pagina da cui proviene l'informazione, così da poterla verificare.

Requisiti decisi durante il lavoro:
- un'interfaccia grafica curata (Streamlit);
- generazione tramite un'**API gratuita (Groq)** invece di un modello eseguito in locale;
- domande **indipendenti**: nessuna memoria della conversazione.

### 1.1 Versione completa e versione semplificata
Il progetto esiste in due versioni, in due cartelle:
- `ChatBot_completo`: due servizi (Gemini e Groq) intercambiabili e memoria della
  conversazione attivabile;
- `ChatBot_semplice` (questa): solo Groq, domande indipendenti.

La versione semplificata è nata dopo le prime prove reali con Gemini, che nel piano
gratuito rispondeva lentamente e restituiva spesso errori 503 (modello sovraccarico).
Togliere il secondo servizio e la memoria riduce il codice da spiegare, le chiamate API per
domanda (sempre una sola) e i possibili punti di errore. Retrieval, verifica delle fonti e
interfaccia sono gli stessi della versione completa.

---

## 2. Punto di partenza

Il progetto parte da un notebook per Google Colab (GPU T4) ricevuto da un compagno di corso.

### 2.1 Cosa faceva

| Fase | Implementazione |
|---|---|
| Lettura | PyMuPDF, testo estratto a blocchi per pagina; i titoli a lettere spaziate ("T E X T") vengono ricompattati |
| Chunking | Pagine brevi intere; pagine lunghe divise in finestre da 450 token con 75 di sovrapposizione, usando gli offset del tokenizer |
| Retrieval semantico | `BAAI/bge-m3` (multilingue), similarità coseno |
| Retrieval lessicale | TF-IDF su unigrammi e bigrammi, con i nomi dei mesi italiani e inglesi normalizzati |
| Fusione | Reciprocal Rank Fusion pesata (0,8 semantico, 0,2 lessicale, k = 60); top-4 passaggi |
| Generazione | `Qwen/Qwen3-4B-Instruct-2507` in fp16 sulla GPU di Colab, decoding greedy |
| Citazioni | L'LLM risponde in JSON `{"risposta", "fonte": "DOC_A"}`; il codice verifica che l'etichetta esista |
| Interfaccia | Ciclo `input()` nel notebook; ogni domanda indipendente |

### 2.2 Punti di forza

- Retrieval ibrido ben costruito: la parte lessicale recupera nomi, sigle e date che gli
  embedding trovano male.
- Chunking basato sui token, con i confini dei token esatti e senza mescolare pagine diverse.
- Citazioni verificate dal codice: il modello non può inventare un numero di pagina.
- Il sistema sa astenersi ("Non ho informazioni sufficienti nei documenti forniti").

### 2.3 Problemi riscontrati

Il codice del notebook è stato eseguito su due PDF di prova, sostituendo soltanto i modelli
pesanti con versioni finte: **27 test superati su 30**. I problemi emersi, insieme a quelli
trovati leggendo il codice:

1. **Parsing del JSON fragile.** Se il modello scrive una frase prima del JSON
   ("Ecco: {...}") o restituisce `"fonte": null` insieme a una risposta valida, la risposta
   va persa e compare "Formato non valido".
2. **Una sola fonte per risposta.** Le domande che richiedono più pagine (es. "Quando si
   presentano i progetti?", pagine 14 e 15) possono citarne una sola.
3. **Valutazione poco probante.** Sul PDF del corso ci sono 18 chunk e ne vengono recuperati 4
   (oltre il 20% del documento). Il risultato di 11/12 è quindi poco significativo, e le prove
   sono scritte sullo stesso documento su cui è stato regolato il prompt.
4. **Chunking mai eseguito davvero.** Il chunk più lungo del PDF del corso ha 322 token, sotto
   la soglia di 450. Inoltre l'`assert` di controllo confrontava con 8192, il limite del
   modello, e non con 450.
5. **Dipendenza dalla GPU.** Qwen3-4B in fp16 occupa circa 8 GB: serve Colab, oppure una GPU
   con più memoria di quella disponibile in locale (RTX 4060, 8 GB).
6. Dettagli minori: import duplicati; ricerca ripetuta nella chat quando la risposta non era
   valida; un numero di citazione (es. "[2]") mostrato con una sola fonte.

---

## 3. Architettura

```
            ┌──────────────── indicizzazione (una volta per PDF, in cache) ────────────────┐
PDF ──► pagine (PyMuPDF) ──► chunk (450 token, overlap 75) ──► embedding BGE-M3 + TF-IDF
                                                                          │
domanda ──────────────────────────────► ricerca ibrida RRF ──► top-k passaggi
                                                                          │
                  LLM Groq, output JSON vincolato ◄───────────────────────┘
                                │
          {"risposta": "... [DOC_A] ...", "fonti": ["DOC_A", "DOC_C"]}
                                │
          verifica e rinumerazione delle citazioni ──► risposta con [1] [2] + file, pagina,
                                                       anteprima della pagina evidenziata
```

| File | Ruolo |
|---|---|
| `rag/pdf.py` | Estrazione del testo per pagina (dal notebook); anteprima della pagina con il passaggio evidenziato |
| `rag/index.py` | Chunking, embedding con cache su disco, TF-IDF, ricerca ibrida RRF (dal notebook, riorganizzato in una classe `Indice`) |
| `rag/llm.py` | Chiamata a Groq con output JSON vincolato e gestione degli errori |
| `rag/chat.py` | Prompt, verifica delle fonti, numerazione delle citazioni |
| `rag/config.py` | Tutti i parametri in un solo punto |
| `app.py` | Interfaccia Streamlit |
| `notebook/ChatBot_RAG.ipynb` | La stessa pipeline passo per passo; importa `rag/`, quindi resta allineato all'app |
| `eval.py` | Le 12 prove del notebook originale (retrieval e risposte) |
| `tests/test_rag.py` | 30 test offline |

La logica è separata dall'interfaccia: `rag/` non dipende da Streamlit e viene usato
identico dall'app, dal notebook, dalla valutazione e dai test.

---

## 4. Modifiche e motivazioni

### 4.1 Generazione via Groq invece di Qwen in locale
- **Cosa:** la risposta è generata da un modello ospitato da Groq, via API gratuita. Il
  retrieval resta interamente locale. Il modello predefinito è `openai/gpt-oss-120b`, ma si
  può scegliere qualunque modello disponibile dal menu, letto dall'API.
- **Perché:**
  - modelli molto più capaci di un 4B;
  - nessuna GPU necessaria per generare, quindi l'app gira su un normale PC;
  - Groq è molto veloce.
- **Alternative considerate:**
  - *Gemini* (Google AI Studio): usato nella versione completa, ma nelle prove reali il
    piano gratuito ha restituito spesso il 503 (modello sovraccarico) e risposte lente;
  - *OpenRouter (modelli `:free`)*: solo 50 richieste al giorno senza credito;
  - *Mistral*: valido, ma serve verificare il numero di telefono.
- **Compromessi:**
  - serve una connessione;
  - il piano gratuito ha limiti di richieste al minuto e al giorno, ma sono per modello:
    se si esauriscono si può cambiare modello.
- Qwen in locale resta un possibile sviluppo futuro (§7).

### 4.2 Ragionamento al minimo
- **Cosa:** per i modelli `gpt-oss`, che ragionano prima di rispondere, la richiesta imposta
  `reasoning_effort="low"`.
- **Perché:** per estrarre una risposta da 4 passaggi non serve un ragionamento lungo, e
  riducendolo la risposta arriva molto prima. È la stessa lezione imparata con Gemini nella
  versione completa, dove il ragionamento predefinito rendeva le risposte lente.

### 4.3 Output strutturato e parsing tollerante
- **Cosa:** lo schema della risposta è definito con Pydantic (`RispostaLLM`) e passato a Groq
  come `json_schema`, con ripiego su `json_object` per i modelli che non lo supportano. Se
  comunque arriva testo intorno al JSON, `estrai_json` recupera il primo oggetto valido. Se
  il formato resta errato, viene fatto un secondo tentativo.
- **Perché:** risolve i due casi fragili del punto 2.3.1, che ora sono coperti da test.

### 4.4 Più fonti per risposta, citazioni nel testo
- **Cosa:** lo schema è `{"risposta": str, "fonti": [str]}`. Il modello cita `[DOC_X]` nel
  testo, accanto all'affermazione. Il codice poi:
  - scarta le etichette inventate;
  - rinumera le citazioni [1], [2], … in ordine di comparsa;
  - associa a ciascuna il suo file e la sua pagina.

  Una risposta senza alcuna fonte valida non viene più buttata via: viene mostrata con un
  avviso e con i passaggi recuperati, perché l'utente possa verificarla.
- **Perché:**
  - le domande che richiedono più pagine o più file ora sono gestite;
  - la citazione accanto a ogni affermazione rende la verifica più semplice.

### 4.5 Domande indipendenti
- **Cosa:** ogni domanda viene cercata nei documenti così com'è, con una sola chiamata
  all'LLM. La chat mostra la cronologia, ma non la usa.
- **Perché:**
  - è il comportamento del notebook di partenza;
  - dimezza le chiamate rispetto alla memoria della versione completa (dove ogni domanda di
    seguito veniva prima riscritta dall'LLM);
  - ogni risposta dipende solo dalla domanda e dai documenti, quindi è più facile da
    spiegare e da valutare.
- **Compromesso:** le domande di seguito ("e la seconda?") non funzionano. Bisogna scrivere
  ogni domanda per intero.

### 4.6 Cache degli embedding e BGE-M3 sulla GPU
- **Cosa:** gli embedding di ogni PDF vengono salvati in `.cache/`. La chiave è l'hash del
  contenuto del file insieme al modello e ai parametri di chunking. BGE-M3 usa la GPU in
  mezza precisione se disponibile (circa 1,1 GB di VRAM), altrimenti la CPU.
- **Perché:**
  - ricaricare lo stesso PDF, anche rinominato, è immediato;
  - se cambiano i parametri, la cache si invalida da sola.

### 4.7 Correzione SSL
- **Cosa:** `truststore.inject_into_ssl()` all'avvio del pacchetto.
- **Perché:** sul PC di sviluppo Python non riusciva a verificare il certificato di
  huggingface.co (`CERTIFICATE_VERIFY_FAILED`), mentre `curl` e `pip` funzionavano. Con
  truststore Python usa l'archivio dei certificati di Windows.

### 4.8 Interfaccia Streamlit
- **Scelta:** Streamlit e non Gradio, per la flessibilità grafica. L'app gira in locale; su
  Colab Streamlit è scomodo, per questo esiste anche il notebook.
- **Elementi principali:**
  - più PDF caricabili insieme, con il numero di pagine e di passaggi per ciascuno e un
    avviso per le pagine senza testo (scansioni);
  - chiave Groq da `.env` oppure inserita nella sessione, e menu dei modelli;
  - impostazioni avanzate (top-k, visualizzazione di tutti i passaggi);
  - stato dell'elaborazione per fasi: ricerca, generazione;
  - citazioni come chip, e per ogni fonte una scheda con l'**immagine della pagina** e il
    passaggio **evidenziato in giallo** (`search_for` + `add_highlight_annot` di PyMuPDF),
    più i punteggi semantico, lessicale e RRF;
  - errori in italiano con il pulsante "Riprova";
  - tema chiaro con un solo colore d'accento e caratteri Inter e Fraunces.
- **Sicurezza:**
  - il testo generato viene mostrato con l'HTML disattivato (escape);
  - il server ascolta solo su `localhost`;
  - la chiave non viene mai scritta nel codice.

### 4.9 Altre correzioni
- `assert` sul chunking rimosso: il controllo reale (≤ 450 token) è ora nei test.
- Le domande vuote vengono rifiutate.
- Timeout di 60 s e fino a 2 nuovi tentativi automatici per gli errori temporanei di Groq.
- Codice organizzato in un pacchetto, con i parametri raccolti in `config.py`.

---

## 5. Prima e dopo

| Aspetto | Notebook di partenza | Versione semplificata |
|---|---|---|
| Esecuzione | Solo Colab con GPU T4 | PC locale (app) oppure Colab/Jupyter (notebook) |
| Generazione | Qwen3-4B locale, fp16 | Groq via API gratuita, modello scelto da un menu |
| Formato risposta | JSON libero, parsing fragile | JSON vincolato da schema + parsing tollerante + secondo tentativo |
| Fonti per risposta | Una | Una o più, citate nel testo e rinumerate |
| Risposta senza fonte | Persa ("Formato non valido") | Mostrata con avviso e passaggi |
| Conversazione | Domande indipendenti | Domande indipendenti (invariato) |
| Interfaccia | `input()` testuale | Streamlit con anteprima delle pagine evidenziate |
| Embedding | Ricalcolati a ogni esecuzione, CPU | Cache su disco, GPU se disponibile |
| Più documenti | Supportati, mai provati | Provati nei test e nell'interfaccia |
| Test | 12 prove di retrieval su un solo PDF | 30 test automatici offline + prove di retrieval e risposte |
| Struttura | Un notebook | Pacchetto `rag/` riusato da app, notebook, valutazione e test |
| Retrieval (BGE-M3 + TF-IDF + RRF) | — | Invariato nella logica e nei parametri |

---

## 6. Valutazione

### 6.1 Test automatici
`tests/test_rag.py` contiene 30 test che non usano modelli né rete: due PDF generati al
volo, un embedder finto e un LLM finto che restituisce uscite scelte dal test. Verificano:
- l'estrazione delle pagine, con lo scarto delle pagine vuote e la numerazione originale;
- il chunking: limite di token, copertura dell'intera pagina, sovrapposizione;
- la cache degli embedding;
- il retrieval su più file e la normalizzazione dei mesi;
- il parsing tollerante del JSON;
- la conversione e la validazione delle citazioni;
- le domande indipendenti, con una sola chiamata all'LLM;
- il ragionamento minimo e lo schema JSON inviati a Groq;
- il rendering della pagina evidenziata.

**Risultato: 30/30 superati**, compresi i due casi che facevano fallire il notebook di
partenza.

### 6.2 Prova dell'interfaccia
L'app è stata provata nel browser con due PDF di prova (programma del corso e regolamento
d'esame):
- caricamento e indicizzazione di più file;
- errore per chiave mancante, con il pulsante "Riprova".

Il notebook è stato eseguito fino alla cella del modello.

### 6.3 Prova con Groq *(da completare)*
Risposte reali con la chiave Groq: correttezza, fonti citate, tempi.

### 6.4 Retrieval sul PDF del corso *(da completare)*
Da eseguire con `eval.py` su `TextMining_2026-27_DayOne.pdf`. Il riferimento è il
notebook originale: 11/12.

---

## 7. Limiti e sviluppi futuri

- **OCR** per i PDF scansionati: oggi le pagine senza testo vengono solo segnalate.
- **Memoria della conversazione**: disponibile nella versione completa (riscrittura della
  domanda con la cronologia).
- **Qwen in locale** come alternativa a Groq (quantizzato a 4 bit per stare negli 8 GB), per
  un funzionamento completamente offline.
- **Reranker** (es. un cross-encoder) sui primi 10–20 risultati, prima di passarne 4 all'LLM.
- **Valutazione più ampia:** più documenti, domande scritte da chi non ha costruito il
  sistema, metriche automatiche di fedeltà della risposta.
- **Pagine "onnipresenti":** nel PDF del corso una pagina di riepilogo compariva in quasi tutti
  i risultati. Rimuovere intestazioni e piè di pagina ripetuti aiuterebbe.

---

## 8. Come eseguire

Vedi `README.md`. In breve, dalla cartella `ChatBot_semplice`:
```powershell
..\.venv\Scripts\streamlit run app.py          # interfaccia
..\.venv\Scripts\python -m pytest -q           # test
..\.venv\Scripts\python eval.py DayOne.pdf     # valutazione
```
