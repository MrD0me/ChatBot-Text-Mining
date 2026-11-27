"""Interfaccia Streamlit del chatbot RAG (versione semplificata: solo Groq, domande indipendenti).

Avvio (dalla cartella di questo file):  ..\\.venv\\Scripts\\streamlit run app.py
"""

import base64
import hashlib
import html
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from rag import config  # noqa: E402  (rag/__init__ sistema i certificati SSL)
from rag.chat import rispondi_rag  # noqa: E402
from rag.index import Indice, carica_embedder, hash_bytes, indicizza_documento  # noqa: E402
from rag.llm import ChiaveNonValida, ErroreLLM, GroqLLM, chiave_da_ambiente, scegli_modello  # noqa: E402
from rag.pdf import render_pagina  # noqa: E402

CARTELLA = Path(__file__).parent
CACHE_EMBEDDINGS = CARTELLA / ".cache" / "embeddings"
ASSETS = CARTELLA / "assets"
NOME_APP = "ATLAS"


def svg(nome: str, suffisso: str = "") -> str:
    """SVG di assets/ su una sola riga (il Markdown non deve vederci righe rientrate o vuote).

    Il suffisso rende unici gli id interni quando lo stesso disegno compare più volte.
    """
    testo = re.sub(r"<!--.*?-->", "", (ASSETS / nome).read_text(encoding="utf-8"), flags=re.S)
    testo = " ".join(riga.strip() for riga in testo.splitlines() if riga.strip())
    return testo.replace('id="atlas-', f'id="atlas{suffisso}-').replace("url(#atlas-", f"url(#atlas{suffisso}-")


st.set_page_config(page_title=f"{NOME_APP} · Esplora i tuoi documenti", page_icon=":material/public:",
                   layout="centered", initial_sidebar_state="expanded")
_onde = base64.b64encode((ASSETS / "onde.svg").read_bytes()).decode()
st.markdown(f'<style>:root{{--atlas-onde:url("data:image/svg+xml;base64,{_onde}")}}'
            f"{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("messaggi", [])  # cronologia mostrata a schermo (le domande restano indipendenti)
ss.setdefault("documenti", {})  # hash del file -> Documento indicizzato
ss.setdefault("indice", None)
ss.setdefault("chiave", "")  # chiave inserita a mano, valida solo per la sessione
ss.setdefault("top_k", config.TOP_K)
ss.setdefault("mostra_passaggi", False)
ss.setdefault("da_inviare", None)  # domanda da un suggerimento o da un nuovo tentativo
ss.setdefault("errore", None)
ss.setdefault("chiave_caricamento", 0)  # cambiarla svuota la casella di caricamento
ss.setdefault("cartella_letta", False)
ss.setdefault("avvisi_documenti", [])  # PDF che non è stato possibile leggere

AVATAR_UTENTE = ":material/person:"
AVATAR_BOT = ":material/public:"

# Suggerimenti del menu con la lampadina nella casella di domanda: etichetta -> domanda inviata.
AZIONI_RAPIDE = {
    ":material/description: Cerca nei documenti": "Di cosa parlano i documenti?",
    ":material/layers: Riassumi i contenuti": "Riassumi i contenuti principali dei documenti.",
    ":material/forum: Rispondi alle domande": "Quali sono i concetti chiave spiegati nei documenti?",
}


# ===================================================================== risorse


@st.cache_resource(show_spinner=False)
def embedder():
    return carica_embedder()


@st.cache_data(show_spinner=False, max_entries=128)
def immagine_pagina(hash_file: str, pagina: int, estratto: str, ritaglio: bool = False) -> bytes:
    documento = next(d for d in ss.documenti.values() if d.hash == hash_file)
    return render_pagina(documento.dati, pagina, estratto, ritaglio=ritaglio)


@st.cache_data(ttl=3600, show_spinner=False)
def modelli_disponibili(impronta_chiave: str, _chiave: str) -> list[str]:
    try:
        return GroqLLM(_chiave, "").elenca_modelli()
    except Exception:
        return []


def chiave_groq() -> str:
    return chiave_da_ambiente() or ss.chiave


# ===================================================================== azioni


def nuova_conversazione():
    ss.messaggi = []
    ss.errore = None


def riprova(domanda: str):
    ss.da_inviare = domanda
    ss.errore = None


def scegli_suggerimento(domanda: str):
    ss.da_inviare = domanda


def ricostruisci_indice() -> None:
    con_testo = [d for d in ss.documenti.values() if d.chunks]
    ss.indice = Indice(con_testo, embedder()) if con_testo else None


def rimuovi_documento(h: str) -> None:
    ss.documenti.pop(h, None)
    ricostruisci_indice()


def aggiorna_documenti(caricati) -> None:
    """Indicizza i PDF appena caricati e li aggiunge ai documenti della sessione.

    I documenti restano in ss.documenti finché non si rimuovono dalla barra laterale; la casella
    di caricamento serve solo ad aggiungerne e dopo ogni caricamento viene svuotata.
    """
    arrivati = {}
    for f in caricati or []:
        dati = f.getvalue()
        arrivati[hash_bytes(dati)] = (f.name, dati)
    # Cartella di PDF caricati all'avvio (opzionale), comoda per demo e verifiche: letta una volta
    # sola, così un documento rimosso non ricompare.
    cartella_pdf = os.environ.get("RAG_CARTELLA_PDF")
    if not ss.cartella_letta:
        ss.cartella_letta = True
        if cartella_pdf and Path(cartella_pdf).is_dir():
            for percorso in sorted(Path(cartella_pdf).glob("*.pdf")):
                dati = percorso.read_bytes()
                arrivati[hash_bytes(dati)] = (percorso.name, dati)
    if not arrivati:
        return

    nuovi = [h for h in arrivati if h not in ss.documenti]
    if nuovi:
        with st.status("Preparo i documenti…", expanded=True) as stato:
            stato.write("Carico il modello di embedding BGE-M3…")
            modello = embedder()
            for h in nuovi:
                nome, dati = arrivati[h]
                stato.write(f"Leggo e indicizzo **{html.escape(nome)}**…")
                try:
                    ss.documenti[h] = indicizza_documento(nome, dati, modello, CACHE_EMBEDDINGS)
                except Exception as e:  # PDF danneggiato o protetto
                    ss.avvisi_documenti.append(f"Non riesco a leggere {nome}: {e.__class__.__name__}.")
            stato.update(label="Documenti pronti", state="complete", expanded=False)
        ricostruisci_indice()
    ss.chiave_caricamento += 1
    st.rerun()  # ridisegna con la casella vuota e il nuovo elenco


# ===================================================================== rendering


def formatta(testo: str) -> str:
    """Testo dell'LLM reso sicuro (niente HTML) con le citazioni [n] come chip."""
    sicuro = html.escape(testo, quote=False)
    return re.sub(r"\[(\d+)\]", r'<span class="cit">\1</span>', sicuro)


def mostra_fonte(fonte: dict, chiave: str) -> None:
    """Pillola della fonte: al clic mostra la pagina originale e il passaggio usato."""
    etichetta = f"Fonte {fonte['id']} · p. {fonte['pagina']}"  # il nome del file è nell'anteprima
    with st.popover(etichetta, icon=":material/description:", key=chiave):
        st.markdown(f'<div class="fonte-titolo"><span class="cit">{fonte["id"]}</span>'
                    f'{html.escape(fonte["file"])} · pagina {fonte["pagina"]}</div>', unsafe_allow_html=True)
        scheda_anteprima, scheda_pagina, scheda_testo = st.tabs(["Anteprima", "Pagina intera", "Testo"])
        documento = next((d for d in ss.documenti.values() if d.nome == fonte["file"]), None)
        for scheda, ritaglio in ((scheda_anteprima, True), (scheda_pagina, False)):
            with scheda:
                if documento is None:
                    st.caption("Il documento è stato rimosso: anteprima non disponibile.")
                else:
                    st.image(immagine_pagina(documento.hash, fonte["pagina"], fonte["testo"], ritaglio),
                             width="stretch")
        st.caption("In giallo il testo passato al modello.")
        with scheda_testo:
            st.markdown(f'<div class="passaggio">{html.escape(fonte["testo"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="punteggi">Fusione RRF {fonte["punteggio"]:.4f} · semantico '
                        f'{fonte["semantico"]:.3f} · lessicale {fonte["lessicale"]:.3f}</div>',
                        unsafe_allow_html=True)


def mostra_risposta(m: dict, n: int) -> None:
    classe = "" if m["trovata"] else ' class="non-so"'
    st.markdown(f"<div{classe}>{formatta(m['testo'])}</div>", unsafe_allow_html=True)
    if m["trovata"] and not m["verificata"]:
        st.warning("Il modello non ha indicato una fonte valida: controlla i passaggi recuperati.",
                   icon=":material/report:")
    if ss.mostra_passaggi or (m["trovata"] and not m["verificata"]):
        with st.expander(f"Tutti i passaggi recuperati ({len(m['passaggi'])})", icon=":material/search:"):
            for i, p in enumerate(m["passaggi"], start=1):
                st.markdown(f"**{i}.** {p['file']} · pagina {p['pagina']} · RRF {p['punteggio']:.4f}")
                st.markdown(f'<div class="passaggio">{html.escape(p["testo"])}</div>', unsafe_allow_html=True)
    # Piede del messaggio: pillole delle fonti a sinistra, modello e tempo a destra.
    with st.container(horizontal=True, key=f"piede_{n}", vertical_alignment="center", gap="small"):
        for k, fonte in enumerate(m["fonti"]):
            mostra_fonte(fonte, f"fonte_{n}_{k}")
        secondi = f"{m['secondi']:.1f}".replace(".", ",")
        modello = html.escape(m["modello"])
        st.markdown(f'<div class="meta">Groq · {modello} · {secondi} s</div>',
                    unsafe_allow_html=True, width="content")


def mostra_errore(errore: dict) -> None:
    with st.chat_message("user", avatar=AVATAR_UTENTE):
        st.markdown(html.escape(errore["domanda"]))
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        st.error(errore["messaggio"], icon=":material/error:")
        colonne = st.columns([1, 3])
        colonne[0].button("Riprova", icon=":material/refresh:", on_click=riprova, args=(errore["domanda"],),
                          width="stretch")


# ===================================================================== impostazioni


def copia_impostazione(nome: str) -> None:
    # I widget della finestra hanno chiavi proprie: quando la finestra è chiusa non vengono
    # disegnati e Streamlit ne cancella lo stato, mentre il valore deve restare.
    ss[nome] = ss[f"_{nome}"]


@st.dialog("Impostazioni", width="medium")
def finestra_impostazioni() -> None:
    st.markdown('<div class="sezione">Ricerca nei documenti</div>', unsafe_allow_html=True)
    st.slider("Passaggi da recuperare", 1, 8, value=ss.top_k, key="_top_k",
              on_change=copia_impostazione, args=("top_k",),
              help="Quanti passaggi (top-k) passare al modello per ogni domanda.")
    st.toggle("Mostra tutti i passaggi recuperati sotto ogni risposta", value=ss.mostra_passaggi,
              key="_mostra_passaggi", on_change=copia_impostazione, args=("mostra_passaggi",))
    st.markdown('<div class="piede">Retrieval locale: BGE-M3 + TF-IDF, fusi con Reciprocal Rank Fusion. '
                'Ogni domanda è indipendente dalle precedenti.</div>', unsafe_allow_html=True)
    if st.button("Fatto", type="primary", width="stretch"):
        st.rerun()  # chiude la finestra e ridisegna la pagina con i nuovi valori


# ===================================================================== casella e comandi sotto la chat


def casella_domanda(testo: str, attiva: bool) -> str | None:
    """Casella di domanda con la lampadina a sinistra, che apre il menu dei suggerimenti."""
    with st.container(key="casella"):
        domanda = st.chat_input(testo, disabled=not attiva, key="domanda")
        with st.popover("", icon=":material/lightbulb:", key="piu", disabled=not attiva, help="Suggerimenti"):
            st.markdown('<div class="sezione primo">Prova a chiedere</div>', unsafe_allow_html=True)
            for i, (etichetta, testo_domanda) in enumerate(AZIONI_RAPIDE.items()):
                st.button(etichetta, key=f"azione_{i}", on_click=scegli_suggerimento, args=(testo_domanda,),
                          width="stretch")
    return domanda


def comandi_chat() -> str:
    """Selettore del modello Groq e "Pulisci chat", in fila sotto la casella. Restituisce il modello."""
    chiave = chiave_groq()
    disponibili = []
    if chiave:
        disponibili = modelli_disponibili(hashlib.sha256(chiave.encode()).hexdigest()[:16], chiave)
    scelte = disponibili or config.GROQ_MODELLI_PREFERITI
    predefinito = scegli_modello(scelte)
    scelto = ss.get("modello")
    etichetta = f"Groq · {scelto if scelto in scelte else predefinito}"
    with st.container(horizontal=True, key="comandi", gap="small", vertical_alignment="center"):
        with st.popover(etichetta, icon=":material/auto_awesome:"):
            if chiave_da_ambiente():
                st.markdown('<div class="stato-chiave">● Chiave caricata dal file .env</div>',
                            unsafe_allow_html=True)
            else:
                ss.chiave = st.text_input(
                    "Chiave API Groq", value=ss.chiave, type="password", placeholder="Incolla qui la chiave…",
                    help="Resta solo in questa sessione. Per non reinserirla, mettila nel file .env.").strip()
                st.caption(f"[Ottieni una chiave gratuita ↗]({config.GROQ_URL_CHIAVE})")
            st.markdown('<div class="sezione">Modello</div>', unsafe_allow_html=True)
            modello = st.selectbox("Modello", scelte, index=scelte.index(predefinito), key="modello",
                                   label_visibility="collapsed")
        st.button("Pulisci chat", icon=":material/delete_sweep:", key="pulisci", on_click=nuova_conversazione,
                  disabled=not ss.messaggi and not ss.errore)
    return modello


# ===================================================================== barra laterale

with st.sidebar:
    st.markdown(f'<div class="marchio">{svg("logo.svg", "-lat")}<div><div class="nome">{NOME_APP}</div>'
                '<div class="sotto">Assistente RAG sui tuoi PDF</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sezione">Carica documenti</div>', unsafe_allow_html=True)
    caricati = st.file_uploader("Carica uno o più PDF", type=["pdf"], accept_multiple_files=True,
                                label_visibility="collapsed", key=f"caricamento_{ss.chiave_caricamento}")

# L'indicizzazione mostra il suo stato nell'area principale, più visibile della barra laterale.
aggiorna_documenti(caricati)

with st.sidebar:
    n = len(ss.documenti)
    st.markdown(f'<div class="sezione">I tuoi documenti<span class="conteggio">{n}</span></div>',
                unsafe_allow_html=True)
    if not ss.documenti:
        st.markdown('<div class="vuoto">Nessun documento ancora. Trascina qui sopra i PDF '
                    'su cui vuoi fare domande.</div>', unsafe_allow_html=True)
    for messaggio in ss.avvisi_documenti:
        st.warning(messaggio, icon=":material/error:")
    ss.avvisi_documenti = []
    for h, d in list(ss.documenti.items()):
        avviso = ""
        if d.pagine_vuote:
            n = len(d.pagine_vuote)
            avviso = (f'<div class="avviso">{n} pagin{"a" if n == 1 else "e"} senza testo '
                      f'(scansioni? serve OCR)</div>')
        nome = html.escape(d.nome)
        with st.container(horizontal=True, key=f"doc_{h}", vertical_alignment="center", gap="small"):
            st.markdown(f'<div class="doc"><div class="icona">PDF</div><div><div class="nome" title="{nome}">'
                        f'{nome}</div><div class="info">{d.n_pagine} pagine · {len(d.chunks)} passaggi</div>'
                        f'{avviso}</div></div>', unsafe_allow_html=True)
            st.button("", icon=":material/delete:", key=f"elimina_{h}", type="tertiary",
                      help=f"Rimuovi {d.nome}", on_click=rimuovi_documento, args=(h,))

    # In fondo alla barra laterale (lo fissa il CSS di .st-key-fondo_laterale).
    with st.container(key="fondo_laterale"):
        if st.button("Impostazioni", icon=":material/settings:", key="apri_impostazioni", width="stretch"):
            finestra_impostazioni()


# ===================================================================== area principale

n_doc = len(ss.indice.documenti) if ss.indice else 0

testo_input = f"Chiedi qualcosa ad {NOME_APP}…" if n_doc else "Carica prima almeno un PDF dalla barra laterale"
intro = None

if not ss.messaggi and not ss.errore:
    # Pagina iniziale: la casella di domanda sta dentro un contenitore, quindi Streamlit la
    # mostra in linea sotto il titolo invece che fissata in fondo alla pagina.
    intro = st.empty()
    with intro.container():
        st.markdown(
            f'<div class="atlas-hero"><div class="atlas-arte">{svg("globo.svg")}'
            f'<div class="atlas-marchio">{svg("logo.svg", "-hero")}<div class="atlas-parola">{NOME_APP}</div>'
            '<div class="atlas-linea"></div></div></div>'
            '<h1>Esplora la tua <span class="sfumato">conoscenza.</span></h1>'
            '<p class="sottotitolo">I tuoi documenti. Un unico assistente intelligente.</p></div>',
            unsafe_allow_html=True)
        domanda = casella_domanda(testo_input, bool(n_doc))
        modello = comandi_chat()
        if not n_doc:
            st.markdown(
                '<div class="passi">'
                '<div class="passo"><span class="n">1</span><div class="t">Carica i PDF</div>'
                '<div class="d">Dalla barra laterale, anche più file insieme. Il testo viene diviso in '
                'passaggi e indicizzato sul tuo computer.</div></div>'
                '<div class="passo"><span class="n">2</span><div class="t">Inserisci la chiave Groq</div>'
                '<div class="d">È gratuita. Mettila nel file .env o incollala nel selettore del modello '
                'sotto la casella.</div></div>'
                '<div class="passo"><span class="n">3</span><div class="t">Fai una domanda</div>'
                '<div class="d">La risposta cita le fonti: aprine una per vedere la pagina originale.</div></div>'
                '</div>', unsafe_allow_html=True)
else:
    # Durante la conversazione la casella torna fissata in fondo alla pagina.
    with st.bottom:  # casella e comandi restano fissati in fondo alla pagina
        domanda = casella_domanda(testo_input, bool(n_doc))
        modello = comandi_chat()
    st.markdown(f'<div class="barra"><span class="titolo">{svg("logo.svg", "-barra")}{NOME_APP}</span>'
                f'<span class="chip">{n_doc} document{"o" if n_doc == 1 else "i"}</span></div>',
                unsafe_allow_html=True)

domanda = (domanda or ss.da_inviare or "").strip()
ss.da_inviare = None
if domanda and intro is not None:
    intro.empty()  # la prima domanda prende il posto della pagina iniziale

for n, m in enumerate(ss.messaggi):
    if m["ruolo"] == "utente":
        with st.chat_message("user", avatar=AVATAR_UTENTE):
            st.markdown(html.escape(m["testo"]))
    else:
        with st.chat_message("assistant", avatar=AVATAR_BOT):
            mostra_risposta(m, n)

if ss.errore:
    mostra_errore(ss.errore)

if domanda and ss.indice is not None:
    ss.errore = None
    with st.chat_message("user", avatar=AVATAR_UTENTE):
        st.markdown(html.escape(domanda))
    ss.messaggi.append({"ruolo": "utente", "testo": domanda})

    errore = None
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        try:
            llm = GroqLLM(chiave_groq(), modello)
            with st.status("Elaboro la domanda…", expanded=False) as stato:
                risposta = rispondi_rag(domanda, ss.indice, llm, top_k=ss.top_k,
                                        avanzamento=lambda t: stato.update(label=t))
                stato.update(label=f"Risposta pronta in {risposta.secondi:.1f} s", state="complete")
            ss.messaggi.append({"ruolo": "assistente", "testo": risposta.testo, "fonti": risposta.fonti,
                                "passaggi": risposta.passaggi, "trovata": risposta.trovata,
                                "verificata": risposta.verificata, "secondi": risposta.secondi,
                                "modello": modello})
        except ChiaveNonValida as e:
            errore = f"{e} Inseriscila nel selettore del modello sotto la casella o nel file .env."
        except ErroreLLM as e:
            errore = str(e)

    if errore:
        ss.messaggi.pop()  # la domanda senza risposta non resta nella chat
        ss.errore = {"messaggio": errore, "domanda": domanda}
    # Ridisegna la pagina: la risposta compare nella cronologia e la barra laterale si aggiorna.
    st.rerun()
