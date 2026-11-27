"""Chunking, embedding e ricerca ibrida (semantica + lessicale) sui PDF."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .pdf import conta_pagine, estrai_pagine

# ---------------------------------------------------------------- embedder


def carica_embedder(nome: str = config.EMBEDDING_MODEL):
    """Carica BGE-M3 sulla GPU se disponibile (in mezza precisione), altrimenti su CPU."""
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        modello = SentenceTransformer(nome, device="cuda")
        modello.half()  # ~1.1 GB di VRAM invece di ~2.2 GB, stessa qualità di ricerca
    else:
        modello = SentenceTransformer(nome, device="cpu")
    return modello


# ---------------------------------------------------------------- chunking


def crea_chunk(pagine: list[dict], tokenizer, max_token: int = config.MAX_TOKEN,
               overlap: int = config.OVERLAP) -> list[dict]:
    """Divide ogni pagina in finestre di token sovrapposte, senza unire pagine diverse.

    Gli offset del tokenizer permettono di tagliare il testo originale esattamente
    sui confini dei token, quindi ogni chunk resta testo leggibile della pagina.
    Una pagina più corta di max_token rimane un chunk unico.
    """
    chunks = []
    for pagina in pagine:
        testo = pagina["testo"]
        posizioni = tokenizer(testo, add_special_tokens=False, return_offsets_mapping=True,
                              truncation=False)["offset_mapping"]
        if not posizioni:
            continue
        for inizio in range(0, len(posizioni), max_token - overlap):
            fine = min(inizio + max_token, len(posizioni))
            pezzo = testo[posizioni[inizio][0]:posizioni[fine - 1][1]]
            chunks.append({**pagina, "testo": pezzo})
            if fine == len(posizioni):
                break
    return chunks


# ---------------------------------------------------------------- lessicale

_MESI_IT = "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre"
_MESI_EN = "january february march april may june july august september october november december"
_ABBR = "jan feb mar apr may jun jul aug sep oct nov dec".split()
MESI = {**dict(zip(_MESI_IT.split(), _ABBR)), **dict(zip(_MESI_EN.split(), _ABBR))}


def tokenizza(testo: str) -> list[str]:
    """Minuscole, senza accenti, mesi italiani e inglesi portati alla stessa forma."""
    testo = unicodedata.normalize("NFKD", testo.lower())
    testo = "".join(c for c in testo if not unicodedata.combining(c))
    return [MESI.get(p, p) for p in re.findall(r"\b\w+\b", testo)]


# ---------------------------------------------------------------- documenti


@dataclass
class Documento:
    """Un PDF indicizzato: pagine, chunk ed embedding dei chunk."""

    nome: str
    hash: str
    dati: bytes = field(repr=False)
    n_pagine: int
    pagine_vuote: list[int]
    chunks: list[dict]
    embeddings: np.ndarray = field(repr=False)


def hash_bytes(dati: bytes) -> str:
    return hashlib.sha256(dati).hexdigest()


def _chiave_cache(hash_file: str) -> str:
    impostazioni = f"{config.EMBEDDING_MODEL}|{config.MAX_TOKEN}|{config.OVERLAP}"
    return hashlib.sha256(f"{hash_file}|{impostazioni}".encode()).hexdigest()[:32]


def indicizza_documento(nome: str, dati: bytes, embedder, cartella_cache: Path | None = None) -> Documento:
    """Estrae, divide in chunk e calcola gli embedding di un PDF.

    Gli embedding vengono salvati su disco, indicizzati per contenuto del file e
    impostazioni di chunking: ricaricare lo stesso PDF non li ricalcola.
    """
    n_pagine = conta_pagine(dati)
    pagine, vuote = estrai_pagine(nome, dati)
    hash_file = hash_bytes(dati)

    if cartella_cache is not None:
        base = Path(cartella_cache) / _chiave_cache(hash_file)
        if base.with_suffix(".npy").exists() and base.with_suffix(".json").exists():
            chunks = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
            for c in chunks:
                c["file"] = nome  # lo stesso file può essere ricaricato con un altro nome
            embeddings = np.load(base.with_suffix(".npy"))
            return Documento(nome, hash_file, dati, n_pagine, vuote, chunks, embeddings)

    chunks = crea_chunk(pagine, embedder.tokenizer)
    if chunks:
        embeddings = embedder.encode([c["testo"] for c in chunks], normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
    else:
        embeddings = np.zeros((0, 1), dtype=np.float32)

    if cartella_cache is not None and chunks:
        Path(cartella_cache).mkdir(parents=True, exist_ok=True)
        np.save(base.with_suffix(".npy"), embeddings)
        base.with_suffix(".json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return Documento(nome, hash_file, dati, n_pagine, vuote, chunks, embeddings)


# ---------------------------------------------------------------- indice


class Indice:
    """Indice ibrido su tutti i PDF caricati.

    - semantico: similarità coseno tra embedding BGE-M3 normalizzati;
    - lessicale: TF-IDF su unigrammi e bigrammi, utile per nomi, sigle e date;
    - fusione: Reciprocal Rank Fusion pesata, che combina le posizioni nelle due
      classifiche invece dei punteggi (che hanno scale diverse).
    """

    def __init__(self, documenti: list[Documento], embedder):
        self.documenti = {d.nome: d for d in documenti}
        self.embedder = embedder
        self.chunks = [c for d in documenti for c in d.chunks]
        if not self.chunks:
            raise ValueError("Nessun testo trovato nei PDF caricati.")
        self.embeddings = np.vstack([d.embeddings for d in documenti if d.chunks])
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizza, token_pattern=None, lowercase=False,
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform([c["testo"] for c in self.chunks])

    def cerca(self, domanda: str, top_k: int = config.TOP_K) -> list[dict]:
        domanda = domanda.strip()
        if not domanda:
            raise ValueError("Scrivi una domanda.")
        if top_k < 1:
            raise ValueError("top_k deve essere almeno 1.")
        query = self.embedder.encode(domanda, normalize_embeddings=True, convert_to_numpy=True)
        semantici = self.embeddings @ np.asarray(query, dtype=np.float32)
        lessicali = (self.tfidf @ self.vectorizer.transform([domanda]).T).toarray().ravel()

        fusi = np.zeros(len(self.chunks))
        for punteggi, peso in [(semantici, config.PESO_SEMANTICO), (lessicali, config.PESO_LESSICALE)]:
            for posizione, i in enumerate(np.argsort(-punteggi, kind="stable"), start=1):
                if punteggi[i] > 0:  # un chunk senza parole in comune non riceve voti lessicali
                    fusi[i] += peso / (config.RRF_K + posizione)

        migliori = np.argsort(-fusi, kind="stable")[:top_k]
        return [{**self.chunks[i], "punteggio": float(fusi[i]),
                 "semantico": float(semantici[i]), "lessicale": float(lessicali[i])}
                for i in migliori]
