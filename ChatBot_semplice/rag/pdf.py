"""Lettura dei PDF e anteprima delle pagine citate."""

import re

import pymupdf

# Titoli scritti con lettere spaziate ("T E X T") diventano parole normali ("TEXT").
_LETTERE_SPAZIATE = re.compile(r"(?<!\w)(?:[A-Z] ){2,}[A-Z](?!\w)")


def _pulisci_riga(riga: str) -> str:
    riga = _LETTERE_SPAZIATE.sub(lambda m: m.group().replace(" ", ""), riga)
    return re.sub(r"[ \t]+", " ", riga).strip()


def estrai_pagine(nome: str, dati: bytes) -> tuple[list[dict], list[int]]:
    """Estrae il testo pagina per pagina, a blocchi, conservando le righe.

    Restituisce le pagine con testo e i numeri delle pagine vuote (ad esempio
    pagine scansionate, che richiederebbero OCR).
    """
    pagine, vuote = [], []
    with pymupdf.open(stream=dati, filetype="pdf") as documento:
        for numero, pagina in enumerate(documento, start=1):
            blocchi = []
            for blocco in pagina.get_text("blocks"):
                if blocco[6] != 0:  # 0 = blocco di testo, 1 = immagine
                    continue
                righe = [_pulisci_riga(r) for r in blocco[4].splitlines()]
                testo = "\n".join(r for r in righe if r)
                if testo:
                    blocchi.append(testo)
            if blocchi:
                pagine.append({"file": nome, "pagina": numero, "testo": "\n\n".join(blocchi)})
            else:
                vuote.append(numero)
    return pagine, vuote


def conta_pagine(dati: bytes) -> int:
    with pymupdf.open(stream=dati, filetype="pdf") as documento:
        return documento.page_count


def render_pagina(dati: bytes, numero: int, estratto: str | None = None,
                  zoom: float = 1.6, ritaglio: bool = False) -> bytes:
    """Immagine PNG della pagina, con le righe dell'estratto evidenziate.

    Con ritaglio=True l'immagine mostra solo la fascia della pagina intorno alle righe
    evidenziate (con un po' di margine): è l'anteprima leggibile delle fonti.
    """
    with pymupdf.open(stream=dati, filetype="pdf") as documento:
        pagina = documento[numero - 1]
        evidenziate = pymupdf.Rect()
        if estratto:
            gia_viste = set()
            for riga in estratto.splitlines():
                riga = riga.strip()
                # Righe troppo corte ("1", "•") evidenzierebbero punti a caso.
                if len(riga) < 4 or riga in gia_viste:
                    continue
                gia_viste.add(riga)
                for area in pagina.search_for(riga[:200]):
                    evidenziazione = pagina.add_highlight_annot(area)
                    evidenziazione.set_colors(stroke=(1.0, 0.83, 0.25))
                    evidenziazione.update(opacity=0.55)
                    evidenziate |= area
        area_immagine = pagina.rect
        if ritaglio and not evidenziate.is_empty:
            margine = 36  # punti, mezzo pollice intorno alle righe evidenziate
            area_immagine = pymupdf.Rect(evidenziate.x0 - margine, evidenziate.y0 - margine,
                                         evidenziate.x1 + margine, evidenziate.y1 + margine) & pagina.rect
        pixmap = pagina.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True, clip=area_immagine)
        return pixmap.tobytes("png")
