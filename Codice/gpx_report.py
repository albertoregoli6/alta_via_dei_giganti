#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpx_report.py
=============
Elabora TUTTE le tracce .gpx presenti in una cartella e per ciascuna genera:
  - un'immagine del percorso sovrapposto a una mappa di sfondo (OpenTopoMap)
  - un grafico del profilo altimetrico (asse X = km percorsi)
  - un report testuale con i dati fondamentali della tappa
  - una riga di riepilogo in un CSV complessivo con tutte le tappe

NON vengono calcolati/mostrati dati di tempo o velocita', come richiesto.

Compatibile con Python 3.6.9.

DIPENDENZE (da installare con pip):
    pip install staticmap matplotlib Pillow requests

USO:
    python3 gpx_report.py /percorso/alla/cartella
    python3 gpx_report.py /percorso/alla/cartella -o /percorso/output
    python3 gpx_report.py /percorso/alla/cartella --no-map      (salta la mappa, utile senza internet)

L'INPUT E' SEMPRE UNA CARTELLA (non un singolo file): lo script cerca tutti i
file con estensione .gpx al suo interno e li elabora uno per uno.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")  # backend senza interfaccia grafica, funziona ovunque
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Libreria opzionale per disegnare la mappa di sfondo.
# Se non e' installata, lo script continua comunque: salta solo l'immagine
# "mappa + percorso" e genera un avviso a schermo.
# --------------------------------------------------------------------------
try:
    from staticmap import StaticMap, Line, CircleMarker
    STATICMAP_DISPONIBILE = True
except ImportError:
    STATICMAP_DISPONIBILE = False

GPX_NS = "http://www.topografix.com/GPX/1/1"

# Sotto questa soglia (in metri) le variazioni di quota vengono considerate
# rumore del GPS e non contano nel calcolo di dislivello positivo/negativo.
SOGLIA_RUMORE_QUOTA = 2.0

# Finestra (in metri) usata per calcolare la pendenza media/massima:
# invece di calcolarla punto-punto (troppo rumorosa), la si calcola ogni
# tot metri di percorso.
FINESTRA_PENDENZA_M = 50.0


# ==========================================================================
# PARSING DEL FILE GPX
# ==========================================================================

def parse_gpx(percorso_file):
    """Legge un file GPX e restituisce (nome_traccia, lista_punti).

    lista_punti e' una lista di tuple (lat, lon, quota) prese da tutti i
    trkpt del file, nell'ordine in cui compaiono (eventuali trk/trkseg
    multipli vengono concatenati in un unico percorso).
    """
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
    """Distanza in metri fra due punti (lat, lon) sulla sfera terrestre."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def calcola_distanze_cumulative(punti):
    """Ritorna la lista delle distanze cumulative (in km) per ogni punto."""
    cum_km = [0.0]
    tot_m = 0.0
    for i in range(1, len(punti)):
        lat1, lon1, _ = punti[i - 1]
        lat2, lon2, _ = punti[i]
        tot_m += distanza_haversine(lat1, lon1, lat2, lon2)
        cum_km.append(tot_m / 1000.0)
    return cum_km


def calcola_dislivelli(quote, soglia=SOGLIA_RUMORE_QUOTA):
    """Calcola dislivello positivo (D+) e negativo (D-) in metri, filtrando
    le micro-oscillazioni dovute al rumore del sensore GPS/barometrico."""
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


def calcola_pendenze(cum_km, quote, finestra_m=FINESTRA_PENDENZA_M):
    """Calcola la pendenza (%) a tratti di 'finestra_m' metri, per ottenere
    pendenza media in salita/discesa e pendenza massima, senza il rumore
    del calcolo punto-punto."""
    pendenze = []
    if len(cum_km) < 2:
        return pendenze
    dist_m = [k * 1000.0 for k in cum_km]
    inizio = 0
    for i in range(1, len(dist_m)):
        if dist_m[i] - dist_m[inizio] >= finestra_m:
            dd = dist_m[i] - dist_m[inizio]
            de = quote[i] - quote[inizio]
            if dd > 0:
                pendenze.append((de / dd) * 100.0)
            inizio = i
    return pendenze


def calcola_statistiche(nome_traccia, punti):
    quote = [p[2] for p in punti if p[2] is not None]
    cum_km = calcola_distanze_cumulative(punti)
    gain, loss = calcola_dislivelli(quote)
    pendenze = calcola_pendenze(cum_km, quote)

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
# GRAFICO ALTIMETRIA
# ==========================================================================

def genera_grafico_altimetria(nome_traccia, cum_km, quote, percorso_output):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(cum_km, quote, color="#b5651d", linewidth=1.6)
    ax.fill_between(cum_km, quote, min(quote), color="#b5651d", alpha=0.22)

    ax.set_xlabel("Distanza percorsa (km)")
    ax.set_ylabel("Quota (m s.l.m.)")
    ax.set_title("Profilo altimetrico - %s" % nome_traccia, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, cum_km[-1] if cum_km else 1)

    plt.tight_layout()
    plt.savefig(percorso_output, dpi=140)
    plt.close(fig)


# ==========================================================================
# IMMAGINE MAPPA + PERCORSO
# ==========================================================================

def genera_mappa(nome_traccia, punti, percorso_output):
    """Genera un'immagine con il percorso disegnato sopra una mappa reale
    (tile OpenTopoMap). Richiede una connessione internet e il pacchetto
    'staticmap'. Se non disponibile, disegna un fallback senza sfondo reale."""

    coords_lonlat = [(lon, lat) for lat, lon, _ in punti]

    if STATICMAP_DISPONIBILE:
        try:
            m = StaticMap(
                900, 700,
                url_template="https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
            )
            m.add_line(Line(coords_lonlat, "#e63946", 4))
            m.add_marker(CircleMarker(coords_lonlat[0], "#2a9d8f", 12))   # partenza
            m.add_marker(CircleMarker(coords_lonlat[-1], "#e63946", 12))  # arrivo
            immagine = m.render()
            immagine.save(percorso_output)
            return
        except Exception as e:
            print("  [!] Impossibile scaricare la mappa di sfondo (%s)." % e)
            print("      Genero comunque un'immagine del percorso senza sfondo cartografico.")

    # Fallback: nessuna mappa di sfondo reale disponibile (libreria mancante
    # o nessuna connessione internet). Disegno solo la traccia.
    lats = [p[0] for p in punti]
    lons = [p[1] for p in punti]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(lons, lats, color="#e63946", linewidth=1.8)
    ax.scatter([lons[0]], [lats[0]], color="#2a9d8f", s=60, zorder=5, label="Partenza")
    ax.scatter([lons[-1]], [lats[-1]], color="#e63946", s=60, zorder=5, label="Arrivo")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.set_title("Percorso (senza sfondo cartografico) - %s" % nome_traccia, fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(percorso_output, dpi=140)
    plt.close(fig)


# ==========================================================================
# REPORT TESTUALE
# ==========================================================================

def scrivi_report_testuale(stats, percorso_output):
    righe = [
        "REPORT TRACCIA: %s" % stats["nome"],
        "=" * 60,
        "Numero di punti GPS:            %d" % stats["numero_punti"],
        "Distanza totale:                %.2f km" % stats["distanza_km"],
        "",
        "Quota di partenza:              %.0f m" % stats["quota_partenza_m"],
        "Quota di arrivo:                %.0f m" % stats["quota_arrivo_m"],
        "Quota minima:                   %.0f m" % stats["quota_min_m"],
        "Quota massima:                  %.0f m" % stats["quota_max_m"],
        "",
        "Dislivello positivo (D+):       %.0f m" % stats["dislivello_positivo_m"],
        "Dislivello negativo (D-):       %.0f m" % stats["dislivello_negativo_m"],
        "",
        "Pendenza media in salita:       %.1f %%" % stats["pendenza_media_salita_pct"],
        "Pendenza media in discesa:      %.1f %%" % stats["pendenza_media_discesa_pct"],
        "Pendenza massima in salita:     %.1f %%" % stats["pendenza_massima_salita_pct"],
        "Pendenza massima in discesa:    %.1f %%" % stats["pendenza_massima_discesa_pct"],
    ]
    with open(percorso_output, "w", encoding="utf-8") as f:
        f.write("\n".join(righe) + "\n")


# ==========================================================================
# UTILITA'
# ==========================================================================

def slug(nome):
    """Trasforma un nome in qualcosa di adatto a un nome di file/cartella."""
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

    genera_grafico_altimetria(nome_traccia, cum_km, quote, os.path.join(cartella_traccia, "altimetria.png"))

    if genera_mappa_bg:
        genera_mappa(nome_traccia, punti, os.path.join(cartella_traccia, "mappa_percorso.png"))

    scrivi_report_testuale(stats, os.path.join(cartella_traccia, "report.txt"))

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
        if not STATICMAP_DISPONIBILE:
            print("[!] Il pacchetto 'staticmap' non e' installato: le mappe di sfondo verranno")
            print("    sostituite da un semplice disegno del percorso (senza cartografia reale).")
            print("    Per averle, esegui:  pip install staticmap\n")

    file_gpx = sorted(glob.glob(os.path.join(cartella_input, "*.gpx")))
    if not file_gpx:
        print("Nessun file .gpx trovato in '%s'." % cartella_input)
        sys.exit(0)

    print("Trovati %d file .gpx da elaborare.\n" % len(file_gpx))

    riepilogo = []
    for percorso_gpx in file_gpx:
        stats = elabora_file(percorso_gpx, cartella_output, genera_mappa_bg=genera_mappa_bg)
        if stats:
            riepilogo.append(stats)
        print("")

    # CSV riassuntivo di tutte le tappe elaborate
    if riepilogo:
        percorso_csv = os.path.join(cartella_output, "riepilogo_tappe.csv")
        campi = [
            "nome", "distanza_km", "quota_partenza_m", "quota_arrivo_m",
            "quota_min_m", "quota_max_m", "dislivello_positivo_m", "dislivello_negativo_m",
            "pendenza_media_salita_pct", "pendenza_media_discesa_pct",
            "pendenza_massima_salita_pct", "pendenza_massima_discesa_pct", "numero_punti",
        ]
        with open(percorso_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campi)
            writer.writeheader()
            for s in riepilogo:
                writer.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
        print("Riepilogo complessivo salvato in: %s" % percorso_csv)

    print("\nFatto. Risultati in: %s" % cartella_output)


if __name__ == "__main__":
    main()
