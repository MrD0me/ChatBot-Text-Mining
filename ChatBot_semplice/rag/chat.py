"""Generazione della risposta: prompt, chiamata all'LLM, verifica delle fonti."""

import re
import string
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from . import config
from .llm import RispostaNonValida


class RispostaLLM(BaseModel):
    """Schema imposto all'LLM: testo della risposta e fonti usate."""

    risposta: str
    fonti: list[str]


SYSTEM_RISPOSTA = (
    "Sei un assistente che risponde in italiano usando esclusivamente i documenti forniti. "
    "Individua le fonti che rispondono direttamente alla domanda. "
    "Preferisci titoli o elenchi espliciti alle descrizioni generali. "
    "Se sono richiesti più elementi, riportali tutti. "
    "Non aggiungere informazioni che non compaiono nei documenti. "
    "Le fonti hanno identificatori DOC_A, DOC_B, DOC_C, ecc.: nel testo cita tra parentesi "
    "quadre la fonte di ogni affermazione, ad esempio [DOC_A] o [DOC_A][DOC_C]. "
    "NON usare numeri di pagina, di lezione o di elenco come identificatori delle fonti. "
    "Restituisci soltanto un JSON del tipo "
    '{"risposta": "testo con citazioni [DOC_A]", "fonti": ["DOC_A"]}, '
    'dove "fonti" elenca tutte e sole le fonti usate. '
    'Se i documenti non contengono la risposta, restituisci {"risposta": "", "fonti": []}.'
)


@dataclass
class Risposta:
    testo: str  # con le citazioni già convertite in [1], [2], ...
    fonti: list[dict]  # passaggi citati, nell'ordine dei numeri [1], [2], ...
    passaggi: list[dict]  # tutti i passaggi recuperati
    trovata: bool = True  # False = "Non ho informazioni sufficienti"
    verificata: bool = True  # False = risposta senza alcuna fonte valida
    secondi: float = 0.0


def etichette_per(n: int) -> list[str]:
    if n > 26:
        raise ValueError("Troppi passaggi: al massimo 26.")
    return [f"DOC_{lettera}" for lettera in string.ascii_uppercase[:n]]


def costruisci_contesto(passaggi: list[dict]) -> str:
    return "\n\n".join(
        f'<source id="{etichetta}">\n{p["testo"]}\n</source>'
        for etichetta, p in zip(etichette_per(len(passaggi)), passaggi)
    )


def _converti_citazioni(testo: str, fonti_dichiarate: list[str], valide: list[str]) -> tuple[str, list[str]]:
    """Sostituisce [DOC_X] con [1], [2], ... nell'ordine di prima comparsa.

    Restituisce il testo convertito e le etichette citate, in ordine. Le etichette
    inventate (non presenti tra i passaggi) vengono scartate.
    """
    nel_testo = re.findall(r"DOC_[A-Z]", testo)
    ordine = []
    for etichetta in nel_testo + list(fonti_dichiarate):
        if etichetta in valide and etichetta not in ordine:
            ordine.append(etichetta)
    numero = {e: i for i, e in enumerate(ordine, start=1)}

    def sostituisci(m: re.Match) -> str:
        etichette = re.findall(r"DOC_[A-Z]", m.group())
        return "".join(f"[{numero[e]}]" for e in etichette if e in numero)

    # [DOC_A], [DOC_A, DOC_B], [DOC_A][DOC_B]
    convertito = re.sub(r"\[\s*DOC_[A-Z](?:\s*[,;]\s*DOC_[A-Z])*\s*\]", sostituisci, testo)
    convertito = re.sub(r"\bDOC_[A-Z]\b", sostituisci, convertito)  # etichette senza parentesi
    convertito = re.sub(r"[ \t]+([.,;:])", r"\1", convertito).strip()

    if ordine and not re.search(r"\[\d+\]", convertito):  # nessuna citazione nel testo
        convertito += " " + "".join(f"[{numero[e]}]" for e in ordine)
    return convertito, ordine


def rispondi_rag(domanda: str, indice, llm, top_k: int = config.TOP_K,
                 avanzamento: Callable[[str], None] | None = None) -> Risposta:
    """Pipeline completa: retrieval → generazione → verifica delle fonti.

    Ogni domanda è indipendente dalle precedenti. `avanzamento`, se indicato, riceve
    una breve descrizione di ogni fase (serve all'interfaccia).
    """
    segnala = avanzamento or (lambda _: None)
    inizio = time.perf_counter()
    domanda = domanda.strip()

    segnala("Cerco i passaggi più pertinenti nei documenti…")
    passaggi = indice.cerca(domanda, top_k)
    etichette = etichette_per(len(passaggi))
    utente = f"DOCUMENTI:\n{costruisci_contesto(passaggi)}\n\nDOMANDA:\n{domanda}"

    segnala("Genero la risposta…")
    dati, ultimo_errore = None, None
    for _ in range(2):  # un secondo tentativo se il formato non è valido
        try:
            dati = RispostaLLM.model_validate(llm.genera_json(SYSTEM_RISPOSTA, utente, RispostaLLM))
            break
        except (RispostaNonValida, ValidationError) as e:
            ultimo_errore = e
    if dati is None:
        raise RispostaNonValida("Il modello ha risposto in un formato non valido, anche al secondo "
                                "tentativo.") from ultimo_errore

    secondi = time.perf_counter() - inizio
    if not dati.risposta.strip():
        return Risposta(config.NON_SO, [], passaggi, trovata=False, secondi=secondi)

    testo, citate = _converti_citazioni(dati.risposta, dati.fonti, etichette)
    fonti = [{"id": n, **passaggi[etichette.index(e)]} for n, e in enumerate(citate, start=1)]
    return Risposta(testo, fonti, passaggi, trovata=True, verificata=bool(fonti), secondi=secondi)
