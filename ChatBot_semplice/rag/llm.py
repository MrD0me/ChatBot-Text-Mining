"""Generazione via API gratuita di Groq.

`GroqLLM.genera_json(system, user, schema)` restituisce un dizionario conforme allo
schema Pydantic richiesto. Gli errori del servizio diventano eccezioni con un
messaggio in italiano.
"""

import json
import os
import re

from pydantic import BaseModel

from . import config


class ErroreLLM(Exception):
    """Errore del servizio, con un messaggio leggibile in italiano."""


class LimiteRaggiunto(ErroreLLM):
    """Il piano gratuito ha esaurito le richieste disponibili (HTTP 429)."""


class ChiaveNonValida(ErroreLLM):
    """Chiave API mancante, errata o senza permessi."""


class RispostaNonValida(ErroreLLM):
    """Il modello ha risposto con un JSON assente o non conforme allo schema."""


def estrai_json(testo: str) -> dict:
    """Recupera il primo oggetto JSON nel testo, anche se è circondato da altro.

    Gestisce i casi visti nei test del notebook di partenza: blocchi ```json``` e
    frasi prima o dopo il JSON ("Ecco la risposta: {...}").
    """
    testo = re.sub(r"^```(?:json)?\s*|\s*```$", "", (testo or "").strip()).strip()
    try:
        dati = json.loads(testo)
        if isinstance(dati, dict):
            return dati
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for inizio in (m.start() for m in re.finditer(r"\{", testo)):
        try:
            dati, _ = decoder.raw_decode(testo[inizio:])
            if isinstance(dati, dict):
                return dati
        except json.JSONDecodeError:
            continue
    raise RispostaNonValida("Il modello non ha restituito un JSON valido.")


def scegli_modello(disponibili: list[str], preferiti: list[str] = config.GROQ_MODELLI_PREFERITI) -> str | None:
    for nome in preferiti:
        if nome in disponibili:
            return nome
    return disponibili[0] if disponibili else None


def chiave_da_ambiente() -> str:
    return os.environ.get(config.GROQ_ENV, "")


class GroqLLM:
    def __init__(self, api_key: str, modello: str):
        from groq import Groq

        if not api_key:
            raise ChiaveNonValida("Manca la chiave API di Groq.")
        self.client = Groq(api_key=api_key, max_retries=2, timeout=60)
        self.modello = modello

    def _chiama(self, system: str, user: str, formato: dict) -> str:
        extra = {}
        if self.modello.startswith("openai/gpt-oss"):
            # Modelli che "ragionano" prima di rispondere: per estrarre la risposta da
            # pochi passaggi basta il ragionamento minimo, che è molto più veloce.
            extra["reasoning_effort"] = "low"
        risposta = self.client.chat.completions.create(
            model=self.modello,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            response_format=formato,
            **extra,
        )
        return risposta.choices[0].message.content or ""

    def genera_json(self, system: str, user: str, schema: type[BaseModel]) -> dict:
        import groq

        schema_json = {"type": "json_schema",
                       "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}}
        try:
            try:
                testo = self._chiama(system, user, schema_json)
            except groq.BadRequestError:
                # Modello senza supporto json_schema, o output non conforme:
                # si ripiega sulla modalità JSON generica, validata poi da noi.
                testo = self._chiama(system, user, {"type": "json_object"})
        except groq.RateLimitError as e:
            raise LimiteRaggiunto("Groq: limite di richieste del piano gratuito raggiunto. "
                                  "Riprova tra poco o scegli un altro modello.") from e
        except (groq.AuthenticationError, groq.PermissionDeniedError) as e:
            raise ChiaveNonValida("Groq: chiave API non valida.") from e
        except groq.NotFoundError as e:
            raise ErroreLLM(f"Groq: il modello «{self.modello}» non è disponibile.") from e
        except groq.APIStatusError as e:
            raise ErroreLLM(f"Groq: errore del servizio (HTTP {e.status_code}). Riprova tra poco "
                            "o scegli un altro modello.") from e
        except groq.APIConnectionError as e:
            raise ErroreLLM("Groq: impossibile contattare il servizio (rete assente o timeout).") from e
        return estrai_json(testo)

    def elenca_modelli(self) -> list[str]:
        esclusi = ("whisper", "tts", "guard", "playai", "orpheus", "compound", "safeguard")
        return sorted(m.id for m in self.client.models.list().data
                      if getattr(m, "active", True) and not any(x in m.id for x in esclusi))
