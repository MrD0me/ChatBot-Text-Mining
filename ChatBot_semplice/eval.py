"""Valutazione sul PDF del corso (TextMining_2026-27_DayOne.pdf).

Riprende le 12 prove del notebook di partenza:
- retrieval: una prova passa se almeno una delle pagine attese è tra i primi k passaggi;
- risposte (opzionale): stampa domanda, risposta attesa e risposta del modello, più tre
  domande senza risposta nel documento, per un confronto manuale tra modelli Groq.

Esempi:
    ..\\.venv\\Scripts\\python eval.py percorso\\TextMining_2026-27_DayOne.pdf
    ..\\.venv\\Scripts\\python eval.py DayOne.pdf --risposte
    ..\\.venv\\Scripts\\python eval.py DayOne.pdf --risposte --modello llama-3.3-70b-versatile
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from rag import config  # noqa: E402
from rag.chat import rispondi_rag  # noqa: E402
from rag.index import Indice, carica_embedder, indicizza_documento  # noqa: E402
from rag.llm import ErroreLLM, GroqLLM, chiave_da_ambiente, scegli_modello  # noqa: E402

PROVE = [
    ("Chi è la docente del corso?", {1, 2, 18}, "Flora Amato."),
    ("Quali sono le tre fasi del corso di Text Mining?", {5},
     "From Deep Learning to LLMs; Decoding models & GenAI services; Orchestration, retrieval & agents."),
    ("Quali sono gli obiettivi del corso?", {6},
     "Capire gli LLM; usarli per problemi testuali; usare server LLM; orchestrare applicazioni."),
    ("Quali conoscenze servono per seguire il corso?", {7}, "Python intermedio e fondamenti di deep learning."),
    ("Quante lezioni sono previste e quanto dura ciascuna?", {12}, "26 lezioni di due ore."),
    ("Cosa è previsto il 10 novembre?", {14}, "Laboratorio: un chatbot RAG su documenti reali."),
    ("Il 12 novembre si fa lezione?", {12, 14}, "No, le lezioni sono sospese."),
    ("Cosa è previsto il 3 dicembre?", {14}, "Pre-assessment primer e avvio del progetto finale."),
    ("Quando si presentano i progetti?", {14, 15}, "15 e 17 dicembre."),
    ("Quali notebook sono associati alla seconda fase?", {5, 10}, "04, 06 e 6.5."),
    ("È necessario avere una propria GPU?", {7, 17}, "No, si usano Colab o endpoint ospitati."),
    ("La prof richiede di saper spiegare il codice del progetto?", {17},
     "Sì, bisogna saper spiegare ogni riga della pipeline."),
]

SENZA_RISPOSTA = [
    "Qual è la password del Wi-Fi dell'aula?",
    "Qual è la data dell'appello di febbraio?",
    "Qual è il peso percentuale del progetto nel voto finale?",
]


def valuta_retrieval(indice: Indice, top_k: int) -> int:
    superate = 0
    print(f"\n=== Retrieval (top-{top_k}) ===")
    for domanda, attese, _ in PROVE:
        recuperate = [r["pagina"] for r in indice.cerca(domanda, top_k)]
        ok = bool(attese & set(recuperate))
        superate += ok
        print(f"{'OK    ' if ok else 'ERRORE'} {domanda}  attese {sorted(attese)} -> {recuperate}")
    print(f"Fonti recuperate: {superate}/{len(PROVE)}")
    return superate


def valuta_risposte(indice: Indice, modello: str | None, top_k: int) -> None:
    chiave = chiave_da_ambiente()
    if not chiave:
        sys.exit(f"Manca {config.GROQ_ENV} nel file .env.")
    if not modello:
        modello = scegli_modello(GroqLLM(chiave, "").elenca_modelli() or config.GROQ_MODELLI_PREFERITI)
    llm = GroqLLM(chiave, modello)
    print(f"\n=== Risposte: Groq · {modello} ===")
    casi = [(q, attesa) for q, _, attesa in PROVE] + [(q, config.NON_SO) for q in SENZA_RISPOSTA]
    for domanda, attesa in casi:
        try:
            r = rispondi_rag(domanda, indice, llm, top_k=top_k)
            fonti = ", ".join(f"[{f['id']}] p.{f['pagina']}" for f in r.fonti) or "-"
            print(f"\nDOMANDA:  {domanda}\nATTESA:   {attesa}\nRISPOSTA: {r.testo}\nFONTI:    {fonti}"
                  f"   ({r.secondi:.1f} s)")
        except ErroreLLM as e:
            print(f"\nDOMANDA:  {domanda}\nERRORE:   {e}")
        time.sleep(1)  # rispetta i limiti al minuto del piano gratuito


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--risposte", action="store_true", help="valuta anche le risposte del modello")
    parser.add_argument("--modello", default=None)
    args = parser.parse_args()

    embedder = carica_embedder()
    documento = indicizza_documento(args.pdf.name, args.pdf.read_bytes(), embedder,
                                    Path(__file__).parent / ".cache" / "embeddings")
    print(f"{documento.nome}: {documento.n_pagine} pagine, {len(documento.chunks)} chunk")
    indice = Indice([documento], embedder)
    valuta_retrieval(indice, args.top_k)
    if args.risposte:
        valuta_risposte(indice, args.modello, args.top_k)


if __name__ == "__main__":
    main()
