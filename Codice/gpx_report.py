#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpx_report.py
=============
Elabora TUTTE le tracce .gpx presenti in una cartella e per ciascuna genera:

  - mappa_percorso.png     percorso su mappa reale (OpenTopoMap):
                              - partenza  = "A" rosso
                              - arrivo    = "B" verde
                              - ogni km   = punto rosso + numero nero
  - altimetria.png         un unico profilo altimetrico con:
                              - colorazione in base alla pendenza
                                (blu = discesa, rosso = salita)
                              - punto di quota massima e minima evidenziati
                              - nomi dei luoghi attraversati (se forniti,
                                vedi sezione "LUOGHI" piu' sotto)
  - dati.md                tabella riassuntiva in formato markdown con i
                            dati fondamentali della tappa (e, se presente,
                            la tabella dei luoghi attraversati)

NON vengono calcolati/mostrati dati di tempo o velocita'.
NON vengono generati: curva ipsografica, dislivello cumulato, dislivello
per km, istogramma pendenze, report.txt, grafico di sinuosita', CSV di
riepilogo tappe.

Compatibile con Python 3.6.9.

--------------------------------------------------------------------------
LUOGHI ATTRAVERSATI (opzionale)
--------------------------------------------------------------------------
Il file GPX contiene solo la traccia GPS (lat/lon/quota): i nomi dei posti
attraversati (rifugi, colli, borgate...) NON sono presenti nel file e non
possono essere estratti automaticamente.

Per farli comparire sul profilo altimetrico, crea un file CSV con lo
STESSO nome del file GPX ma con suffisso "_luoghi.csv", nella stessa
cartella. Esempio, per:
    Tappa_6__Da_Gressoney-Saint-Jean_a_Crest.gpx
crea:
    Tappa_6__Da_Gressoney-Saint-Jean_a_Crest_luoghi.csv
con questo formato (intestazione obbligatoria):
    nome,km
    Gressoney-Saint-Jean,0
    Tschemenoal,1.1
    Alpenzu,2.6
    Col Pinter,7.7
    Crest,12.5

Se questo file non esiste, lo script elabora comunque tutto il resto: il
profilo altimetrico viene generato normalmente (colorazione per pendenza,
max/min), semplicemente SENZA nomi di luoghi, dato che non e' possibile
inventarli o dedurli in modo affidabile dalla sola traccia GPS.

--------------------------------------------------------------------------
DIPENDENZE (pip install -r requirements.txt):
    matplotlib
    numpy
    Pillow
    requests

USO:
    python3 gpx_report.py /percorso/alla/cartella
    python3 gpx_report.py /percorso/alla/cartella -o /percorso/output
    python3 gpx_report.py /percorso/alla/cartella --no-map

L'INPUT E' SEMPRE UNA CARTELLA (non un singolo file): lo script cerca tutti
i file con estensione .gpx al suo interno e li elabora uno per uno.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    import requests
    from PIL import Image, ImageDraw, ImageFont
    MAPPA_DISPONIBILE = True
except ImportError:
    MAPPA_DISPONIBILE = False

GPX_NS = "http://www.topografix.com/GPX/1/1"

# Sotto questa soglia (in metri) le variazioni di quota sono considerate
# rumore del GPS e non contano nel calcolo di dislivello positivo/negativo.
SOGLIA_RUMORE_QUOTA = 2.0

# Finestra (in metri) per calcolare la pendenza a tratti (statistiche).
FINESTRA_PENDENZA_M = 50.0

# Parametri mappa
TILE_SIZE = 256
MAPPA_LARGHEZZA = 900
MAPPA_ALTEZZA = 700
MAPPA_PADDING_PX = 60
MAPPA_URL_TEMPLATE = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
MAPPA_SUBDOMAINS = ["a", "b", "c"]
MAPPA_USER_AGENT = "gpx-report-script/2.0 (uso personale, escursionismo)"
MAPPA_ZOOM_MASSIMO = 17

COLORE_A_PARTENZA = (211, 47, 47)     # rosso
COLORE_B_ARRIVO = (56, 142, 60)       # verde
COLORE_KM_INTERMEDIO = (211, 47, 47)  # rosso (punto)
COLORE_KM_TESTO = (0, 0, 0)           # nero (numero)


# ==========================================================================
# PARSING DEL FILE GPX
# ==========================================================================

def parse_gpx(percorso_file):
    """Legge un file GPX e restituisce (nome_traccia, lista_punti).
    lista_punti e' una lista di tuple (lat, lon, quota)."""
    tree = ET.parse(percorso_file)
    root = tree.getroot()

    nome_el = root.find("{%s}metadata/{%s}name" % (GPX_NS, GPX_NS))
    if nome_el is None or not nome_el.text:
        nome_el = root.find(".//{%s}trk/{%s}name" % (GPX_NS, GPX_NS))
    nome_traccia = nome_el.text.strip() if (nome_el is not None and nome_el.text) else os.path.splitext(os.path.basename(percorso_file))[0]

    punti = []
    for trkpt in root.iter("{%s}trkpt" % GPX_NS):
        lat = float(trkpt.get("lat"))
        lon = float(trkpt.get("lon"))
        ele_el = trkpt.find("{%s}ele" % GPX_NS)
        quota = float(ele_el.text) if ele_el is not None and ele_el.text is not None else None
        punti.append((lat, lon, quota))

    return nome_traccia, punti


# ==========================================================================
# CALCOLI GEOGRAFICI E STATISTICHE
# ==========================================================================

def distanza_haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def calcola_distanze_cumulative(punti):
    cum_km = [0.0]
    tot_m = 0.0
    for i in range(1, len(punti)):
        lat1, lon1, _ = punti[i - 1]
        lat2, lon2, _ = punti[i]
        tot_m += distanza_haversine(lat1, lon1, lat2, lon2)
        cum_km.append(tot_m / 1000.0)
    return cum_km


def calcola_dislivelli(quote, soglia=SOGLIA_RUMORE_QUOTA):
    if not quote:
        return 0.0, 0.0
    gain = 0.0
    loss = 0.0
    quota_rif = quote[0]
    for q in quote[1:]:
        delta = q - quota_rif
        if abs(delta) >= soglia:
            if delta > 0:
                gain += delta
            else:
                loss += -delta
            quota_rif = q
    return gain, loss


def calcola_segmenti_pendenza(cum_km, quote, finestra_m=FINESTRA_PENDENZA_M):
    """Pendenza (%) a tratti di 'finestra_m' metri: lista di (lunghezza_m, pendenza_pct)."""
    segmenti = []
    if len(cum_km) < 2:
        return segmenti
    dist_m = [k * 1000.0 for k in cum_km]
    inizio = 0
    for i in range(1, len(dist_m)):
        if dist_m[i] - dist_m[inizio] >= finestra_m:
            dd = dist_m[i] - dist_m[inizio]
            de = quote[i] - quote[inizio]
            if dd > 0:
                segmenti.append((dd, (de / dd) * 100.0))
            inizio = i
    return segmenti


def calcola_pendenza_smussata_per_punto(cum_km, quote, mezza_finestra_punti=4):
    """Pendenza locale (%) per la colorazione del profilo (media mobile)."""
    n = len(quote)
    grad = [0.0] * n
    for i in range(n):
        lo = max(0, i - mezza_finestra_punti)
        hi = min(n - 1, i + mezza_finestra_punti)
        dd = (cum_km[hi] - cum_km[lo]) * 1000.0
        de = quote[hi] - quote[lo]
        grad[i] = (de / dd * 100.0) if dd > 0 else 0.0
    return grad


def calcola_statistiche(nome_traccia, punti):
    quote = [p[2] for p in punti if p[2] is not None]
    cum_km = calcola_distanze_cumulative(punti)
    gain, loss = calcola_dislivelli(quote)
    segmenti = calcola_segmenti_pendenza(cum_km, quote)
    pendenze = [p for _, p in segmenti]

    pendenze_salita = [p for p in pendenze if p > 0]
    pendenze_discesa = [p for p in pendenze if p < 0]

    stats = {
        "nome": nome_traccia,
        "numero_punti": len(punti),
        "distanza_km": cum_km[-1] if cum_km else 0.0,
        "quota_partenza_m": quote[0] if quote else None,
        "quota_arrivo_m": quote[-1] if quote else None,
        "quota_min_m": min(quote) if quote else None,
        "quota_max_m": max(quote) if quote else None,
        "dislivello_positivo_m": gain,
        "dislivello_negativo_m": loss,
        "pendenza_media_salita_pct": (sum(pendenze_salita) / len(pendenze_salita)) if pendenze_salita else 0.0,
        "pendenza_media_discesa_pct": (sum(pendenze_discesa) / len(pendenze_discesa)) if pendenze_discesa else 0.0,
        "pendenza_massima_salita_pct": max(pendenze) if pendenze else 0.0,
        "pendenza_massima_discesa_pct": min(pendenze) if pendenze else 0.0,
    }
    return stats, cum_km, quote


# ==========================================================================
# LUOGHI ATTRAVERSATI (sidecar CSV opzionale + auto-suggerimento)
# ==========================================================================

def percorso_luoghi_csv(percorso_gpx):
    base = os.path.splitext(percorso_gpx)[0]
    return base + "_luoghi.csv"


def leggi_luoghi(percorso_gpx, cum_km, quote):
    """Legge il CSV 'nome,km' accanto al GPX, se esiste. Ritorna una lista
    di dict {nome, km, quota} (quota interpolata dal profilo)."""
    percorso_csv = percorso_luoghi_csv(percorso_gpx)
    if not os.path.isfile(percorso_csv):
        return None

    luoghi = []
    with open(percorso_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for riga in reader:
            try:
                nome = riga["nome"].strip()
                km = float(riga["km"])
            except (KeyError, ValueError, AttributeError):
                continue
            quota_interp = float(np.interp(km, cum_km, quote))
            luoghi.append({"nome": nome, "km": km, "quota": quota_interp})
    return luoghi


# ==========================================================================
# GRAFICO: PROFILO ALTIMETRICO (pendenza + max/min + luoghi)
# ==========================================================================

def genera_grafico_altimetria(nome_traccia, cum_km, quote, luoghi, percorso_output):
    grad = calcola_pendenza_smussata_per_punto(cum_km, quote)

    punti_xy = np.array([cum_km, quote]).T.reshape(-1, 1, 2)
    segmenti = np.concatenate([punti_xy[:-1], punti_xy[1:]], axis=1)

    norm = plt.Normalize(-30, 30)
    lc = LineCollection(segmenti, cmap="RdYlBu_r", norm=norm, linewidth=3)
    lc.set_array(np.array(grad))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.add_collection(lc)
    ax.fill_between(cum_km, quote, min(quote), color="gray", alpha=0.08)

    q_min, q_max = min(quote), max(quote)
    margine = (q_max - q_min) * 0.15 + 20

    # --- luoghi attraversati (etichette in stile "scheda tappa") ---
    if luoghi:
        for i, l in enumerate(luoghi):
            ax.axvline(l["km"], color="#888888", linestyle=":", linewidth=0.8, ymax=0.92)
            ax.scatter([l["km"]], [l["quota"]], color="#333333", s=30, zorder=6, edgecolor="white", linewidth=0.8)
            offset_su = (i % 2 == 0)
            y_testo = q_max + margine * (0.55 if offset_su else 0.95)
            ax.annotate(
                "%s\n%.0f m" % (l["nome"], l["quota"]),
                (l["km"], y_testo), ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="#222222",
            )

    # --- massimo e minimo ---
    i_max = int(np.argmax(quote))
    i_min = int(np.argmin(quote))

    ax.scatter([cum_km[i_max]], [quote[i_max]], color="#d62828", zorder=7, s=55, edgecolor="white", linewidth=1)
    ax.annotate(
        "MAX %.0f m (%.1f km)" % (quote[i_max], cum_km[i_max]),
        (cum_km[i_max], quote[i_max]), textcoords="offset points", xytext=(0, 10),
        ha="center", fontsize=8, fontweight="bold", color="#d62828",
    )

    ax.scatter([cum_km[i_min]], [quote[i_min]], color="#1d3557", zorder=7, s=55, edgecolor="white", linewidth=1)
    ax.annotate(
        "MIN %.0f m (%.1f km)" % (quote[i_min], cum_km[i_min]),
        (cum_km[i_min], quote[i_min]), textcoords="offset points", xytext=(0, -18),
        ha="center", fontsize=8, fontweight="bold", color="#1d3557",
    )

    ax.set_xlim(0, cum_km[-1] if cum_km else 1)
    ax.set_ylim(q_min - margine * 0.3, q_max + margine * 1.5)

    cbar = fig.colorbar(lc, ax=ax, pad=0.015)
    cbar.set_label("Pendenza (%) - blu = discesa, rosso = salita")

    ax.set_xlabel("Distanza percorsa (km)")
    ax.set_ylabel("Quota (m s.l.m.)")
    ax.set_title("Profilo altimetrico - %s" % nome_traccia, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(percorso_output, dpi=150)
    plt.close(fig)


# ==========================================================================
# MAPPA: percorso + A (rosso) + B (verde) + tacche km (rosso/nero)
# ==========================================================================

def _lonlat_to_world_px(lon, lat, zoom):
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    return x, y


def _scegli_zoom(lats, lons, img_w, img_h, padding_px, max_zoom=MAPPA_ZOOM_MASSIMO):
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    for zoom in range(max_zoom, 0, -1):
        x1, y1 = _lonlat_to_world_px(min_lon, max_lat, zoom)
        x2, y2 = _lonlat_to_world_px(max_lon, min_lat, zoom)
        if abs(x2 - x1) <= (img_w - 2 * padding_px) and abs(y2 - y1) <= (img_h - 2 * padding_px):
            return zoom
    return 1


def _scarica_tile(tx, ty, zoom, session, indice_subdomain):
    sub = MAPPA_SUBDOMAINS[indice_subdomain % len(MAPPA_SUBDOMAINS)]
    url = MAPPA_URL_TEMPLATE.format(s=sub, z=zoom, x=tx, y=ty)
    risposta = session.get(url, headers={"User-Agent": MAPPA_USER_AGENT}, timeout=10)
    risposta.raise_for_status()
    from io import BytesIO
    return Image.open(BytesIO(risposta.content)).convert("RGB")


def _disegna_marker(draw, xy, testo, colore_rgb):
    r = 14
    x, y = xy
    draw.ellipse([x - r, y - r, x + r, y + r], fill=colore_rgb, outline=(255, 255, 255), width=2)
    draw.text((x - 4, y - 7), testo, fill=(255, 255, 255))


def genera_mappa(nome_traccia, punti, cum_km, percorso_output):
    """Percorso su mappa reale (OpenTopoMap): A=rosso (partenza), B=verde
    (arrivo), tacca rossa + numero nero a ogni km. Richiede internet."""
    lats = [p[0] for p in punti]
    lons = [p[1] for p in punti]

    if MAPPA_DISPONIBILE:
        try:
            zoom = _scegli_zoom(lats, lons, MAPPA_LARGHEZZA, MAPPA_ALTEZZA, MAPPA_PADDING_PX)
            centro_lon = (min(lons) + max(lons)) / 2.0
            centro_lat = (min(lats) + max(lats)) / 2.0
            center_x, center_y = _lonlat_to_world_px(centro_lon, centro_lat, zoom)

            canvas = Image.new("RGB", (MAPPA_LARGHEZZA, MAPPA_ALTEZZA), (230, 230, 225))

            tile_x0 = int((center_x - MAPPA_LARGHEZZA / 2) // TILE_SIZE)
            tile_x1 = int((center_x + MAPPA_LARGHEZZA / 2) // TILE_SIZE)
            tile_y0 = int((center_y - MAPPA_ALTEZZA / 2) // TILE_SIZE)
            tile_y1 = int((center_y + MAPPA_ALTEZZA / 2) // TILE_SIZE)

            session = requests.Session()
            i_sub = 0
            for tx in range(tile_x0, tile_x1 + 1):
                for ty in range(tile_y0, tile_y1 + 1):
                    try:
                        tile_img = _scarica_tile(tx, ty, zoom, session, i_sub)
                    except Exception:
                        tile_img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (235, 235, 230))
                    i_sub += 1
                    px = tx * TILE_SIZE - (center_x - MAPPA_LARGHEZZA / 2)
                    py = ty * TILE_SIZE - (center_y - MAPPA_ALTEZZA / 2)
                    canvas.paste(tile_img, (int(round(px)), int(round(py))))

            def geo_to_canvas(lat, lon):
                x, y = _lonlat_to_world_px(lon, lat, zoom)
                return (x - (center_x - MAPPA_LARGHEZZA / 2), y - (center_y - MAPPA_ALTEZZA / 2))

            draw = ImageDraw.Draw(canvas)
            percorso_px = [geo_to_canvas(lat, lon) for lat, lon in zip(lats, lons)]
            draw.line(percorso_px, fill=(60, 60, 60), width=4, joint="curve")

            if cum_km:
                n_km = int(cum_km[-1])
                prossimo_km = 1
                for i in range(len(cum_km)):
                    if cum_km[i] >= prossimo_km:
                        x, y = percorso_px[i]
                        r = 7
                        draw.ellipse([x - r, y - r, x + r, y + r], fill=COLORE_KM_INTERMEDIO, outline=(255, 255, 255), width=1)
                        draw.text((x + 9, y - 6), str(prossimo_km), fill=COLORE_KM_TESTO)
                        prossimo_km += 1
                        if prossimo_km > n_km:
                            break

            _disegna_marker(draw, percorso_px[0], "A", COLORE_A_PARTENZA)
            _disegna_marker(draw, percorso_px[-1], "B", COLORE_B_ARRIVO)

            canvas.save(percorso_output)
            return
        except Exception as e:
            print("  [!] Impossibile scaricare la mappa di sfondo (%s)." % e)
            print("      Genero comunque un'immagine del percorso senza sfondo cartografico.")

    _genera_mappa_fallback(nome_traccia, punti, cum_km, percorso_output)


def _genera_mappa_fallback(nome_traccia, punti, cum_km, percorso_output):
    lats = [p[0] for p in punti]
    lons = [p[1] for p in punti]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(lons, lats, color="#3c3c3c", linewidth=1.8, zorder=2)

    r, g, b = COLORE_A_PARTENZA
    ax.scatter([lons[0]], [lats[0]], color=(r / 255, g / 255, b / 255), s=140, zorder=5)
    ax.annotate("A", (lons[0], lats[0]), color="white", ha="center", va="center", fontweight="bold", zorder=6)

    r, g, b = COLORE_B_ARRIVO
    ax.scatter([lons[-1]], [lats[-1]], color=(r / 255, g / 255, b / 255), s=140, zorder=5)
    ax.annotate("B", (lons[-1], lats[-1]), color="white", ha="center", va="center", fontweight="bold", zorder=6)

    if cum_km:
        n_km = int(cum_km[-1])
        prossimo_km = 1
        for i in range(len(cum_km)):
            if cum_km[i] >= prossimo_km:
                ax.scatter([lons[i]], [lats[i]], color="#d32f2f", s=45, zorder=4)
                ax.annotate(str(prossimo_km), (lons[i], lats[i]), textcoords="offset points",
                            xytext=(6, 4), fontsize=7, color="black", fontweight="bold")
                prossimo_km += 1
                if prossimo_km > n_km:
                    break

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.set_title("Percorso (senza sfondo cartografico) - %s" % nome_traccia, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(percorso_output, dpi=140)
    plt.close(fig)


# ==========================================================================
# TABELLA DATI IN MARKDOWN
# ==========================================================================

def scrivi_tabella_markdown(stats, luoghi, percorso_output):
    righe = []
    righe.append("# %s" % stats["nome"])
    righe.append("")
    righe.append("| Dato | Valore |")
    righe.append("|---|---|")
    righe.append("| Distanza totale | %.2f km |" % stats["distanza_km"])
    righe.append("| Quota di partenza | %.0f m |" % stats["quota_partenza_m"])
    righe.append("| Quota di arrivo | %.0f m |" % stats["quota_arrivo_m"])
    righe.append("| Quota minima | %.0f m |" % stats["quota_min_m"])
    righe.append("| Quota massima | %.0f m |" % stats["quota_max_m"])
    righe.append("| Dislivello positivo (D+) | %.0f m |" % stats["dislivello_positivo_m"])
    righe.append("| Dislivello negativo (D-) | %.0f m |" % stats["dislivello_negativo_m"])
    righe.append("| Pendenza media in salita | %.1f %% |" % stats["pendenza_media_salita_pct"])
    righe.append("| Pendenza media in discesa | %.1f %% |" % stats["pendenza_media_discesa_pct"])
    righe.append("| Pendenza massima in salita | %.1f %% |" % stats["pendenza_massima_salita_pct"])
    righe.append("| Pendenza massima in discesa | %.1f %% |" % stats["pendenza_massima_discesa_pct"])
    righe.append("| Numero di punti GPS | %d |" % stats["numero_punti"])

    if luoghi:
        righe.append("")
        righe.append("## Luoghi attraversati")
        righe.append("")
        righe.append("| Luogo | Km | Quota |")
        righe.append("|---|---|---|")
        for l in luoghi:
            righe.append("| %s | %.1f km | %.0f m |" % (l["nome"], l["km"], l["quota"]))

    with open(percorso_output, "w", encoding="utf-8") as f:
        f.write("\n".join(righe) + "\n")


# ==========================================================================
# UTILITA'
# ==========================================================================

def slug(nome):
    nome = nome.strip().lower()
    nome = re.sub(r"[àá]", "a", nome)
    nome = re.sub(r"[èé]", "e", nome)
    nome = re.sub(r"[ìí]", "i", nome)
    nome = re.sub(r"[òó]", "o", nome)
    nome = re.sub(r"[ùú]", "u", nome)
    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    return nome.strip("_")[:80] or "traccia"


# ==========================================================================
# ELABORAZIONE DI UN SINGOLO FILE
# ==========================================================================

def elabora_file(percorso_gpx, cartella_output, genera_mappa_bg=True):
    nome_file = os.path.basename(percorso_gpx)
    print("Elaboro: %s" % nome_file)

    try:
        nome_traccia, punti = parse_gpx(percorso_gpx)
    except ET.ParseError as e:
        print("  [!] File non valido, salto: %s" % e)
        return None

    if len(punti) < 2:
        print("  [!] Traccia con meno di 2 punti, salto.")
        return None

    stats, cum_km, quote = calcola_statistiche(nome_traccia, punti)

    cartella_traccia = os.path.join(cartella_output, slug(os.path.splitext(nome_file)[0]))
    os.makedirs(cartella_traccia, exist_ok=True)

    luoghi = leggi_luoghi(percorso_gpx, cum_km, quote)
    if luoghi is None:
        print("  [i] Nessun file '_luoghi.csv' trovato: il profilo non avra' nomi di luoghi.")
        luoghi = []

    genera_grafico_altimetria(nome_traccia, cum_km, quote, luoghi, os.path.join(cartella_traccia, "altimetria.png"))

    if genera_mappa_bg:
        genera_mappa(nome_traccia, punti, cum_km, os.path.join(cartella_traccia, "mappa_percorso.png"))

    scrivi_tabella_markdown(stats, luoghi, os.path.join(cartella_traccia, "dati.md"))

    print("  -> risultati salvati in: %s" % cartella_traccia)
    return stats


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estrae dati e genera mappa + profilo altimetrico per tutte le tracce GPX in una cartella."
    )
    parser.add_argument("cartella_input", help="Cartella contenente i file .gpx da elaborare")
    parser.add_argument("-o", "--output", dest="cartella_output", default=None,
                         help="Cartella dove salvare i risultati (default: sottocartella 'output' dentro la cartella di input)")
    parser.add_argument("--no-map", dest="no_map", action="store_true",
                         help="Salta la generazione dell'immagine mappa+percorso (utile senza connessione internet)")
    args = parser.parse_args()

    cartella_input = args.cartella_input
    if not os.path.isdir(cartella_input):
        print("Errore: '%s' non e' una cartella valida." % cartella_input)
        sys.exit(1)

    cartella_output = args.cartella_output or os.path.join(cartella_input, "output")
    os.makedirs(cartella_output, exist_ok=True)

    if args.no_map:
        genera_mappa_bg = False
    else:
        genera_mappa_bg = True
        if not MAPPA_DISPONIBILE:
            print("[!] I pacchetti 'requests'/'Pillow' non sono installati: le mappe di sfondo")
            print("    verranno sostituite da un semplice disegno del percorso (senza cartografia reale).")
            print("    Per averle, esegui:  pip install requests Pillow\n")

    file_gpx = sorted(glob.glob(os.path.join(cartella_input, "*.gpx")))
    if not file_gpx:
        print("Nessun file .gpx trovato in '%s'." % cartella_input)
        sys.exit(0)

    print("Trovati %d file .gpx da elaborare.\n" % len(file_gpx))

    for percorso_gpx in file_gpx:
        elabora_file(percorso_gpx, cartella_output, genera_mappa_bg=genera_mappa_bg)
        print("")

    print("Fatto. Risultati in: %s" % cartella_output)


if __name__ == "__main__":
    main()