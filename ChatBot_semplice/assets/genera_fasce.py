"""Genera il rilievo topografico del globo di vetro e lo inserisce in globo.svg.

Uso:  python genera_fasce.py globo.svg

Il rilievo nasce da due "campi di altezza" (uno blu, uno pesca) costruiti con gaussiane e
rumore deformato; le curve di livello si estraggono con un marching squares scritto qui
sotto (serve solo numpy). Ogni livello diventa uno strato di carta ritagliata: i livelli
si disegnano dal basso verso l'alto e il filtro atlas-g-strato aggiunge ombra e filo di luce.
Forma e colori si cambiano nella sezione PARAMETRI.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

# ============================================================== PARAMETRI
PASSO = 3.0                      # risoluzione della griglia (px del viewBox)
X0, X1, Y0, Y1 = 128, 594, 8, 352
SEME = 7

# blu: dal livello più basso (bordo esterno) al più alto (cime)
COLORI_BLU = ["#a9cefb", "#7cb4f8", "#4f96f0", "#2f7ce8", "#1f68dc", "#1656cf",
              "#1048bf", "#2a70e0", "#5c9cf2", "#9cc6fa"]
# pesca: dal bordo verso le cime
COLORI_PESCA = ["#fee6cb", "#fcd2a6", "#f9bc82", "#f4a564", "#f8c48f", "#fde0c0"]


# ============================================================== campi di altezza
ys = np.arange(Y0, Y1 + PASSO, PASSO)
xs = np.arange(X0, X1 + PASSO, PASSO)
X, Y = np.meshgrid(xs, ys)
rng = np.random.default_rng(SEME)


def onde(Xc, Yc, n=10, lmin=60, lmax=200):
    """Somma di onde con direzioni e lunghezze casuali, normalizzata in [-1, 1] circa."""
    tot = np.zeros_like(Xc)
    for _ in range(n):
        ang = rng.uniform(0, math.pi)
        lung = rng.uniform(lmin, lmax)
        fase = rng.uniform(0, 2 * math.pi)
        tot += np.sin((Xc * math.cos(ang) + Yc * math.sin(ang)) * 2 * math.pi / lung + fase)
    return tot / math.sqrt(n)


def gauss(Xc, Yc, cx, cy, s, sy=None):
    sy = sy or s
    return np.exp(-(((Xc - cx) / s) ** 2 + ((Yc - cy) / sy) ** 2) / 2)


# deformazione del dominio: rende le forme sinuose e organiche (bordi a "fiordo")
Xw = X + 34 * onde(X, Y, 8, 160, 320) + 12 * onde(X, Y, 8, 55, 110)
Yw = Y + 34 * onde(X, Y, 8, 160, 320) + 12 * onde(X, Y, 8, 55, 110)

# masse blu: un continente in alto a sinistra, una cresta scura al centro, uno a destra in basso
blu = (0.80 * gauss(Xw, Yw, 240, 125, 85, 70)
       + 0.75 * gauss(Xw, Yw, 350, 60, 55, 80)
       + 0.55 * gauss(Xw, Yw, 300, 205, 65)
       + 0.75 * gauss(Xw, Yw, 530, 250, 75, 65)
       + 0.55 * gauss(Xw, Yw, 185, 240, 60)
       + 0.20 * onde(Xw, Yw, 10, 70, 160))

# il "fiume" pesca: scende dal centro in alto verso destra
punti_pesca = [(420, 0), (440, 45), (465, 90), (495, 130), (530, 165), (565, 195)]
pesca = np.max([gauss(Xw, Yw, px, py, 44) for px, py in punti_pesca], axis=0)
pesca = pesca + 0.16 * onde(Xw, Yw, 10, 70, 160) - 0.1


# ============================================================== marching squares
def contorni(F: np.ndarray, t: float) -> list[list[tuple[float, float]]]:
    """Curve chiuse dove F == t (la griglia va bordata con valori bassi)."""
    alto = F > t
    ny, nx = F.shape

    def punto(chiave):
        tipo, i, j = chiave
        if tipo == "h":   # tra (i,j) e (i,j+1)
            a, b = F[i, j], F[i, j + 1]
            f = (t - a) / (b - a)
            return xs_p[j] + f * PASSO, ys_p[i]
        a, b = F[i, j], F[i + 1, j]  # "v": tra (i,j) e (i+1,j)
        f = (t - a) / (b - a)
        return xs_p[j], ys_p[i] + f * PASSO

    segmenti = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            caso = (alto[i, j] << 3) | (alto[i, j + 1] << 2) | (alto[i + 1, j + 1] << 1) | alto[i + 1, j]
            if caso in (0, 15):
                continue
            su, dx, giu, sx = ("h", i, j), ("v", i, j + 1), ("h", i + 1, j), ("v", i, j)
            centro_alto = (F[i, j] + F[i, j + 1] + F[i + 1, j] + F[i + 1, j + 1]) / 4 > t
            tabella = {
                1: [(sx, giu)], 2: [(giu, dx)], 3: [(sx, dx)], 4: [(su, dx)],
                5: [(sx, su), (giu, dx)] if centro_alto else [(su, dx), (sx, giu)],
                6: [(su, giu)], 7: [(sx, su)], 8: [(sx, su)], 9: [(su, giu)],
                10: [(su, dx), (sx, giu)] if centro_alto else [(sx, su), (giu, dx)],
                11: [(su, dx)], 12: [(sx, dx)], 13: [(giu, dx)], 14: [(sx, giu)],
            }
            segmenti.extend(tabella[caso])

    vicini: dict = {}
    for a, b in segmenti:
        vicini.setdefault(a, []).append(b)
        vicini.setdefault(b, []).append(a)
    visti, anelli = set(), []
    for inizio in vicini:
        if inizio in visti:
            continue
        anello, prec, cur = [], None, inizio
        while cur not in visti:
            visti.add(cur)
            anello.append(punto(cur))
            prossimi = [v for v in vicini[cur] if v != prec and v not in visti]
            if not prossimi:
                break
            prec, cur = cur, prossimi[0]
        if len(anello) > 6:
            anelli.append(anello)
    return anelli


def chaikin(p, giri=2):
    for _ in range(giri):
        n = len(p)
        q = []
        for k in range(n):
            a, b = p[k], p[(k + 1) % n]
            q.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
            q.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
        p = q
    return p


def semplifica(p, eps=0.45):
    """Ramer-Douglas-Peucker."""
    if len(p) < 3:
        return p
    a, b = np.array(p[0]), np.array(p[-1])
    ab = b - a
    lung = np.hypot(*ab) or 1e-9
    d = [abs(ab[0] * (a[1] - q[1]) - ab[1] * (a[0] - q[0])) / lung for q in p]
    k = int(np.argmax(d))
    if d[k] > eps:
        return semplifica(p[:k + 1], eps)[:-1] + semplifica(p[k:], eps)
    return [p[0], p[-1]]


def area(p):
    return abs(sum(p[k][0] * p[(k + 1) % len(p)][1] - p[(k + 1) % len(p)][0] * p[k][1]
                   for k in range(len(p)))) / 2


def percorso(F, t):
    parti = []
    for anello in contorni(F, t):
        if area(anello) < 160:
            continue
        liscio = chaikin(anello)
        # anello chiuso: si spezza nel punto più lontano dall'inizio e si semplificano le due metà
        a = np.array(liscio[0])
        k = int(np.argmax([np.hypot(*(np.array(q) - a)) for q in liscio]))
        liscio = semplifica(liscio[:k + 1])[:-1] + semplifica(liscio[k:] + [liscio[0]])[:-1]
        parti.append("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in liscio) + "Z")
    return "".join(parti)


# bordo basso attorno alla griglia: tutte le curve diventano chiuse
xs_p = np.concatenate([[xs[0] - PASSO], xs, [xs[-1] + PASSO]])
ys_p = np.concatenate([[ys[0] - PASSO], ys, [ys[-1] + PASSO]])
blu_p = np.pad(blu, 1, constant_values=-9)
pesca_p = np.pad(pesca, 1, constant_values=-9)

strati = []
livelli_blu = [0.20, 0.27, 0.34, 0.41, 0.48, 0.55, 0.63, 0.80, 0.95, 1.08]
for t, colore in zip(livelli_blu, COLORI_BLU):
    d = percorso(blu_p, t)
    if d:
        strati.append(f'<path d="{d}" fill="{colore}" fill-rule="evenodd" filter="url(#atlas-g-strato)"/>')
livelli_pesca = [0.22, 0.30, 0.38, 0.46, 0.62, 0.78]
for t, colore in zip(livelli_pesca, COLORI_PESCA):
    d = percorso(pesca_p, t)
    if d:
        strati.append(f'<path d="{d}" fill="{colore}" fill-rule="evenodd" filter="url(#atlas-g-strato)"/>')

percorso_svg = Path(sys.argv[1])
testo = percorso_svg.read_text(encoding="utf-8")
testo = re.sub(r"<!-- FASCE:GRADIENTI -->.*?<!-- /FASCE:GRADIENTI -->",
               "<!-- FASCE:GRADIENTI --><!-- /FASCE:GRADIENTI -->", testo, flags=re.S)
testo = re.sub(r"<!-- FASCE:STRATI -->.*?<!-- /FASCE:STRATI -->",
               "<!-- FASCE:STRATI -->\n      " + "\n      ".join(strati) + "\n      <!-- /FASCE:STRATI -->",
               testo, flags=re.S)
percorso_svg.write_text(testo, encoding="utf-8")
print(f"strati: {len(strati)}  dimensione svg: {len(testo) / 1024:.0f} KB")
