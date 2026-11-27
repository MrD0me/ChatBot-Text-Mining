"""Test offline della pipeline RAG.

Non scaricano modelli e non chiamano API: usano due PDF generati al volo, un
embedder finto (bag of words con hashing) e un LLM finto che restituisce risposte
scelte dal test. Così si verifica la logica del codice, non la qualità dei modelli.

Eseguire con:  .venv\\Scripts\\python -m pytest -q
"""

import hashlib
import re

import numpy as np
import pymupdf
import pytest

from rag import config
from rag.chat import RispostaLLM, _converti_citazioni, rispondi_rag
from rag.index import Indice, crea_chunk, indicizza_documento, tokenizza
from rag.llm import GroqLLM, RispostaNonValida, estrai_json
from rag.pdf import estrai_pagine, render_pagina

# ------------------------------------------------------------------ fixture


class TokenizerFinto:
    """Un token per parola o simbolo, con gli offset come un tokenizer vero."""

    def __call__(self, testo, add_special_tokens=False, return_offsets_mapping=False, truncation=False):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", testo)]
        return {"offset_mapping": offsets, "input_ids": list(range(len(offsets)))}


class EmbedderFinto:
    tokenizer = TokenizerFinto()

    def encode(self, testi, **kwargs):
        singolo = isinstance(testi, str)
        vettori = []
        for t in [testi] if singolo else testi:
            v = np.zeros(512, dtype=np.float32)
            for w in tokenizza(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
            vettori.append(v / (np.linalg.norm(v) or 1))
        vettori = np.array(vettori)
        return vettori[0] if singolo else vettori


def crea_pdf(pagine: list[str]) -> bytes:
    doc = pymupdf.open()
    for testo in pagine:
        pagina = doc.new_page()
        if testo:
            pagina.insert_textbox(pymupdf.Rect(40, 40, 560, 800), testo, fontsize=6)
    dati = doc.tobytes()
    doc.close()
    return dati


FILLER = ("Questa sezione descrive argomenti generali di elaborazione del linguaggio naturale, "
          "reti neurali, tokenizzazione e modelli statistici del testo. ")
LUNGA = FILLER * 25 + "Il progetto finale vale il quaranta per cento del voto complessivo."


@pytest.fixture(scope="module")
def pdf_corso():
    return crea_pdf([
        "T E X T  M I N I N G\nDocente del corso: Flora Amato",
        LUNGA,
        "",
        "Il 10 novembre si svolge il laboratorio sul chatbot RAG.",
        "Le reti ricorrenti LSTM gestiscono sequenze lunghe.",
        "I transformer usano il meccanismo di attenzione.",
    ])


@pytest.fixture(scope="module")
def pdf_regolamento():
    return crea_pdf([
        "Regolamento esami.\nL'appello invernale si tiene il 12 February 2027 in aula T3.",
        "La biblioteca del dipartimento apre alle ore 9 e chiude alle 19.",
    ])


@pytest.fixture(scope="module")
def indice(pdf_corso, pdf_regolamento, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    emb = EmbedderFinto()
    docs = [indicizza_documento("corso.pdf", pdf_corso, emb, cache),
            indicizza_documento("regolamento.pdf", pdf_regolamento, emb, cache)]
    return Indice(docs, emb)


class LLMProgrammato:
    """Restituisce in sequenza le uscite indicate; registra i prompt ricevuti."""

    nome, modello = "Test", "programmato"

    def __init__(self, *uscite):
        self.uscite = list(uscite)
        self.prompt = []

    def genera_json(self, system, user, schema):
        self.prompt.append((system, user))
        uscita = self.uscite.pop(0)
        if isinstance(uscita, Exception):
            raise uscita
        return estrai_json(uscita) if isinstance(uscita, str) else uscita


# ------------------------------------------------------------------ PDF


def test_estrazione_pagine(pdf_corso):
    pagine, vuote = estrai_pagine("corso.pdf", pdf_corso)
    assert [p["pagina"] for p in pagine] == [1, 2, 4, 5, 6]
    assert vuote == [3]
    assert "TEXT MINING" in pagine[0]["testo"]


def test_render_pagina_con_evidenziazione(pdf_regolamento):
    semplice = render_pagina(pdf_regolamento, 2)
    evidenziata = render_pagina(pdf_regolamento, 2, "La biblioteca del dipartimento apre alle ore 9")
    assert semplice[:8] == b"\x89PNG\r\n\x1a\n"
    assert evidenziata != semplice  # l'evidenziazione cambia l'immagine


# ------------------------------------------------------------------ chunking


def test_chunking_pagina_lunga(pdf_corso):
    pagine, _ = estrai_pagine("corso.pdf", pdf_corso)
    tok = TokenizerFinto()
    lunga = next(p for p in pagine if p["pagina"] == 2)
    chunks = crea_chunk([lunga], tok)
    lunghezze = [len(tok(c["testo"])["offset_mapping"]) for c in chunks]
    assert len(chunks) > 1
    assert max(lunghezze) <= config.MAX_TOKEN
    assert all(c["pagina"] == 2 for c in chunks)
    assert lunga["testo"].startswith(chunks[0]["testo"][:50])
    assert lunga["testo"].endswith(chunks[-1]["testo"][-50:])
    for a, b in zip(chunks, chunks[1:]):  # sovrapposizione tra chunk consecutivi
        assert a["testo"][-60:] in b["testo"]


def test_pagina_corta_resta_intera(pdf_corso):
    pagine, _ = estrai_pagine("corso.pdf", pdf_corso)
    corte = [p for p in pagine if p["pagina"] != 2]
    assert len(crea_chunk(corte, TokenizerFinto())) == len(corte)


def test_cache_embeddings(pdf_regolamento, tmp_path):
    class Contatore(EmbedderFinto):
        chiamate = 0

        def encode(self, testi, **kw):
            if not isinstance(testi, str):
                Contatore.chiamate += 1
            return super().encode(testi, **kw)

    emb = Contatore()
    a = indicizza_documento("a.pdf", pdf_regolamento, emb, tmp_path)
    b = indicizza_documento("copia.pdf", pdf_regolamento, emb, tmp_path)
    assert Contatore.chiamate == 1  # il secondo caricamento usa la cache
    assert np.allclose(a.embeddings, b.embeddings)
    assert all(c["file"] == "copia.pdf" for c in b.chunks)


# ------------------------------------------------------------------ retrieval


@pytest.mark.parametrize("domanda, attesa", [
    ("Quanto vale il progetto finale nel voto?", ("corso.pdf", 2)),
    ("Chi è la docente del corso?", ("corso.pdf", 1)),
    ("Quando è l'appello del 12 febbraio?", ("regolamento.pdf", 1)),
    ("A che ora apre la biblioteca?", ("regolamento.pdf", 2)),
    ("Cosa si fa il 10 novembre?", ("corso.pdf", 4)),
])
def test_retrieval_su_piu_file(indice, domanda, attesa):
    trovati = [(r["file"], r["pagina"]) for r in indice.cerca(domanda)]
    assert attesa in trovati


def test_mesi_normalizzati():
    assert tokenizza("12 febbraio") == tokenizza("12 February")


@pytest.mark.parametrize("domanda", ["", "   "])
def test_domanda_vuota(indice, domanda):
    with pytest.raises(ValueError):
        indice.cerca(domanda)


# ------------------------------------------------------------------ parsing JSON


@pytest.mark.parametrize("testo", [
    '{"risposta": "Alle 9.", "fonti": ["DOC_A"]}',
    '```json\n{"risposta": "Alle 9.", "fonti": ["DOC_A"]}\n```',
    'Ecco: {"risposta": "Alle 9.", "fonti": ["DOC_A"]} spero aiuti',  # falliva nel notebook
])
def test_estrai_json_tollerante(testo):
    assert estrai_json(testo)["risposta"] == "Alle 9."


def test_estrai_json_assente():
    with pytest.raises(RispostaNonValida):
        estrai_json("Alle 9 di mattina.")


# ------------------------------------------------------------------ citazioni


def test_citazioni_multiple_e_rinumerate():
    valide = ["DOC_A", "DOC_B", "DOC_C", "DOC_D"]
    testo, ordine = _converti_citazioni("Primo [DOC_C]. Secondo [DOC_A, DOC_C].", ["DOC_C", "DOC_A"], valide)
    assert testo == "Primo [1]. Secondo [2][1]."
    assert ordine == ["DOC_C", "DOC_A"]


def test_citazione_inventata_scartata():
    testo, ordine = _converti_citazioni("Vero [DOC_A] e falso [DOC_Z].", ["DOC_A", "DOC_Z"], ["DOC_A", "DOC_B"])
    assert ordine == ["DOC_A"]
    assert "DOC_Z" not in testo


def test_citazioni_mancanti_nel_testo():
    testo, _ = _converti_citazioni("Alle 9.", ["DOC_B"], ["DOC_A", "DOC_B"])
    assert testo == "Alle 9. [1]"


# ------------------------------------------------------------------ pipeline


DOMANDA = "A che ora apre la biblioteca?"


def test_risposta_con_fonte_giusta(indice):
    llm = LLMProgrammato('{"risposta": "Apre alle 9 [DOC_A].", "fonti": ["DOC_A"]}')
    r = rispondi_rag(DOMANDA, indice, llm)
    assert r.trovata and r.verificata
    assert r.testo == "Apre alle 9 [1]."
    assert (r.fonti[0]["file"], r.fonti[0]["pagina"]) == ("regolamento.pdf", 2)
    assert 'id="DOC_D"' in llm.prompt[0][1]  # quattro passaggi nel contesto


def test_piu_fonti_da_file_diversi(indice):
    llm = LLMProgrammato('{"risposta": "Biblioteca [DOC_A], appello [DOC_B].", "fonti": ["DOC_A", "DOC_B"]}')
    r = rispondi_rag("Biblioteca e appello invernale", indice, llm)
    assert len(r.fonti) == 2
    assert [f["id"] for f in r.fonti] == [1, 2]


def test_risposta_vuota_non_so(indice):
    r = rispondi_rag(DOMANDA, indice, LLMProgrammato({"risposta": "", "fonti": []}))
    assert not r.trovata and r.testo == config.NON_SO and r.fonti == []


def test_risposta_senza_fonte_valida_segnalata(indice):
    # Nel notebook "fonte": null faceva perdere la risposta; ora resta, ma non verificata.
    r = rispondi_rag(DOMANDA, indice, LLMProgrammato({"risposta": "Alle 9.", "fonti": []}))
    assert r.trovata and not r.verificata and r.testo == "Alle 9."


def test_secondo_tentativo_se_formato_non_valido(indice):
    llm = LLMProgrammato("niente json", '{"risposta": "Alle 9 [DOC_A]", "fonti": ["DOC_A"]}')
    assert rispondi_rag(DOMANDA, indice, llm).verificata
    assert len(llm.prompt) == 2


def test_formato_non_valido_due_volte(indice):
    with pytest.raises(RispostaNonValida):
        rispondi_rag(DOMANDA, indice, LLMProgrammato("no", {"altro": 1}))


# ------------------------------------------------------------------ Groq


class _Messaggio:
    def __init__(self, contenuto):
        self.choices = [type("C", (), {"message": type("M", (), {"content": contenuto})()})()]


@pytest.mark.parametrize("modello, atteso", [("openai/gpt-oss-120b", "low"),
                                            ("llama-3.3-70b-versatile", None)])
def test_groq_ragionamento_minimo(monkeypatch, modello, atteso):
    llm = GroqLLM("chiave-finta", modello)  # nessuna chiamata di rete
    ricevuti = {}

    def finto_create(**kwargs):
        ricevuti.update(kwargs)
        return _Messaggio('{"risposta": "ok", "fonti": []}')

    monkeypatch.setattr(llm.client.chat.completions, "create", finto_create)
    assert llm.genera_json("s", "u", RispostaLLM) == {"risposta": "ok", "fonti": []}
    assert ricevuti.get("reasoning_effort") == atteso
    assert ricevuti["response_format"]["type"] == "json_schema"


def test_domande_indipendenti(indice):
    # Versione semplificata: una sola chiamata all'LLM, la domanda è usata così com'è.
    llm = LLMProgrammato({"risposta": "", "fonti": []})
    rispondi_rag("e quando chiude?", indice, llm)
    assert len(llm.prompt) == 1
    assert llm.prompt[0][1].endswith("DOMANDA:\ne quando chiude?")


def test_schema_risposta():
    assert set(RispostaLLM.model_json_schema()["required"]) == {"risposta", "fonti"}
