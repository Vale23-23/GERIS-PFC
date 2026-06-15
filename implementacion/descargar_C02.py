#!/usr/bin/env python3
"""
Descargador de archivos NetCDF desde http://164.73.222.53/imagenes/C02/
Satélite GOES-19, banda C02 (Canal Rojo visible)
"""

import os
import re
import sys
import time
import argparse
import requests
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "http://164.73.222.53/imagenes/C02/"
DESTINO   = "./descargas_C02"

def listar_archivos(url, extension=".nc", timeout=30):
    """Scrapea el índice Apache y devuelve lista de URLs de archivos."""
    print(f"📡 Conectando a {url} ...")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Error al acceder al servidor: {e}")
        sys.exit(1)

    # Busca todos los href que terminen en la extensión pedida
    patron = re.compile(r'href="([^"]+' + re.escape(extension) + r')"')
    archivos = patron.findall(resp.text)

    urls = [urljoin(url, nombre) for nombre in archivos]
    print(f"✅ Se encontraron {len(urls)} archivos {extension}")
    return urls


def descargar_archivo(url, destino_dir, reintentos=3, pausa=2):
    """Descarga un archivo con soporte de reintentos."""
    nombre = url.split("/")[-1]
    ruta   = Path(destino_dir) / nombre

    # Saltar si ya está descargado y no está vacío
    if ruta.exists() and ruta.stat().st_size > 0:
        print(f"  ⏭  Ya existe: {nombre}")
        return True

    for intento in range(1, reintentos + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                descargado = 0

                with open(ruta, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):  # 256 KB
                        f.write(chunk)
                        descargado += len(chunk)

                # Verificación básica de tamaño
                if total and descargado < total * 0.99:
                    raise IOError(f"Descarga incompleta ({descargado}/{total} bytes)")

            mb = descargado / 1_048_576
            print(f"  ✔  {nombre}  ({mb:.1f} MB)")
            return True

        except Exception as e:
            print(f"  ⚠  Intento {intento}/{reintentos} fallido: {e}")
            if ruta.exists():
                ruta.unlink()          # borra archivo parcial
            if intento < reintentos:
                time.sleep(pausa * intento)

    print(f"  ❌ No se pudo descargar: {nombre}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Descarga archivos .nc del directorio GOES-19 C02"
    )
    parser.add_argument(
        "--url", default=BASE_URL,
        help="URL base del directorio (default: %(default)s)"
    )
    parser.add_argument(
        "--destino", default=DESTINO,
        help="Carpeta local de destino (default: %(default)s)"
    )
    parser.add_argument(
        "--ext", default=".nc",
        help="Extensión de archivos a descargar (default: %(default)s)"
    )
    parser.add_argument(
        "--limite", type=int, default=0,
        help="Descargar solo los primeros N archivos (0 = todos)"
    )
    parser.add_argument(
        "--reintentos", type=int, default=3,
        help="Número de reintentos por archivo (default: %(default)s)"
    )
    args = parser.parse_args()

    # Crear carpeta destino
    Path(args.destino).mkdir(parents=True, exist_ok=True)
    print(f"📁 Destino: {Path(args.destino).resolve()}\n")

    # Obtener lista de archivos
    urls = listar_archivos(args.url, extension=args.ext)
    if args.limite:
        urls = urls[: args.limite]
        print(f"ℹ️  Modo limitado: descargando solo {args.limite} archivos\n")

    # Descargar
    ok = 0
    fail = 0
    t0 = time.time()

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}]", end=" ")
        if descargar_archivo(url, args.destino, reintentos=args.reintentos):
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"✅ Descargados : {ok}")
    print(f"❌ Fallidos    : {fail}")
    print(f"⏱  Tiempo total: {elapsed/60:.1f} min")
    print(f"📁 Archivos en : {Path(args.destino).resolve()}")


if __name__ == "__main__":
    main()
