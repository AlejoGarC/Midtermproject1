import shutil
import subprocess
import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CARPETA_PROYECTO = Path(__file__).resolve().parent


CARPETA_ENTRADAS = CARPETA_PROYECTO / "entradas"

CARPETA_SALIDAS = CARPETA_PROYECTO / "salidas"
CARPETA_SALIDAS.mkdir(parents=True, exist_ok=True)

RUTA_IMAGEN = str(CARPETA_ENTRADAS / "foto.webp")
RUTA_AUDIO = str(CARPETA_ENTRADAS / "Audio.mp3")

RUTA_VIDEO_MUDO = str(CARPETA_SALIDAS / "salida_sin_audio.mp4")
RUTA_VIDEO_FINAL = str(CARPETA_SALIDAS / "visual_equalizer.mp4")
RUTA_FIGURA = str(CARPETA_SALIDAS / "bandas_normalizadas.png")

INICIO_S = 0.0
DURACION_S = 20.0
FPS = 30

SR = 22050
HOP = 512
N_FFT = 2048
N_BANDAS = 12

FILAS, COLUMNAS = 3, 4
ANCHO, ALTO = 1280, 720

PERCENTIL_MIN, PERCENTIL_MAX = 5, 95
GAMMA = 1.3
T_ATAQUE = 0.03
T_RELEASE = 0.25
T_ATAQUE_ONSET = 0.01
T_RELEASE_ONSET = 0.12

BRILLO_MIN = 0.35
BRILLO_MAX = 1.60
ZOOM_MAX = 0.10
FLASH_GAIN = 0.35
BORDE_PX = 3


def cargar_audio(ruta, sr=SR, inicio=INICIO_S, duracion=DURACION_S):
    y, sr = librosa.load(ruta, sr=sr, mono=True, offset=inicio, duration=duracion)
    dur = len(y) / sr
    if dur < 10:
        print(f"AVISO: solo hay {dur:.1f} s de audio; el enunciado pide 10-20 s de video.")
    return y, sr, dur


def energias_por_banda(y, sr, n_bandas=N_BANDAS):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=n_bandas
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=60)
    t_audio = librosa.times_like(mel_db[0], sr=sr, hop_length=HOP)
    return mel_db, t_audio


def normalizar_bandas(mel_db):
    lo = np.percentile(mel_db, PERCENTIL_MIN, axis=1, keepdims=True)
    hi = np.percentile(mel_db, PERCENTIL_MAX, axis=1, keepdims=True)
    norm = np.clip((mel_db - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return norm ** GAMMA


def suavizar_ataque_release(x, dt, t_ataque, t_release):
    a_att = 1.0 - np.exp(-dt / t_ataque)
    a_rel = 1.0 - np.exp(-dt / t_release)
    y = np.zeros_like(x, dtype=np.float64)
    y[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        prev = y[..., t - 1]
        cur = x[..., t]
        alpha = np.where(cur > prev, a_att, a_rel)
        y[..., t] = prev + alpha * (cur - prev)
    return y


def fuerza_de_golpes(y, sr):
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    return np.clip(onset / (np.percentile(onset, 95) + 1e-9), 0.0, 1.0)


def interpolar_a_video(datos, t_audio, t_video):
    datos = np.atleast_2d(datos)
    return np.array([np.interp(t_video, t_audio, fila) for fila in datos])


def preparar_imagen(ruta, ancho, alto):
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No pude leer la imagen: {ruta}")
    h, w = img.shape[:2]
    aspecto = ancho / alto
    if w / h > aspecto:
        nw = int(h * aspecto)
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:
        nh = int(w / aspecto)
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh, :]
    return cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_AREA)


def cortar_tiles(img, filas, columnas):
    th, tw = img.shape[0] // filas, img.shape[1] // columnas
    tiles = [
        [img[r * th:(r + 1) * th, c * tw:(c + 1) * tw].copy() for c in range(columnas)]
        for r in range(filas)
    ]
    return tiles, th, tw


def banda_a_tile(b, filas, columnas):
    fila = filas - 1 - b // columnas
    col = b % columnas
    return fila, col


def transformar_tile(tile, energia, flash):
    ganancia = BRILLO_MIN + (BRILLO_MAX - BRILLO_MIN) * energia + FLASH_GAIN * flash
    zoom = 1.0 + ZOOM_MAX * energia

    h, w = tile.shape[:2]
    if zoom > 1.001:
        ch, cw = int(h / zoom), int(w / zoom)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        tile = cv2.resize(tile[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.convertScaleAbs(tile, alpha=float(ganancia), beta=0)


def renderizar_frame(tiles, th, tw, filas, columnas, energias_frame, flash):
    frame = np.zeros((filas * th, columnas * tw, 3), dtype=np.uint8)
    for b, energia in enumerate(energias_frame):
        r, c = banda_a_tile(b, filas, columnas)
        y0, x0 = r * th, c * tw
        frame[y0:y0 + th, x0:x0 + tw] = transformar_tile(tiles[r][c], energia, flash)
        cv2.rectangle(frame, (x0, y0), (x0 + tw - 1, y0 + th - 1), (0, 0, 0), BORDE_PX)
    return frame


def escribir_video(ruta, tiles, th, tw, filas, columnas, energias_v, flash_v, fps):
    n_frames = energias_v.shape[1]
    out = cv2.VideoWriter(
        ruta, cv2.VideoWriter_fourcc(*"mp4v"), fps, (columnas * tw, filas * th)
    )
    if not out.isOpened():
        raise RuntimeError("No se pudo abrir el VideoWriter (revisa la ruta y el códec).")
    for i in range(n_frames):
        out.write(renderizar_frame(tiles, th, tw, filas, columnas, energias_v[:, i], flash_v[i]))
        if i % 100 == 0:
            print(f"  frame {i}/{n_frames}")
    out.release()


def buscar_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def unir_audio(video_mudo, ruta_audio, salida, inicio, duracion):
    ffmpeg = buscar_ffmpeg()
    if ffmpeg is None:
        print("No encontré ffmpeg: el video queda SIN audio. Instala ffmpeg o "
              "'pip install imageio-ffmpeg', o únelos con cualquier editor.")
        return False
    cmd = [
        ffmpeg, "-y", "-i", video_mudo,
        "-ss", str(inicio), "-i", ruta_audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-t", f"{duracion:.3f}", salida,
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        print("ffmpeg falló:\n", resultado.stderr[-500:])
        return False
    return True


def guardar_figura(mel_db, norm, suave, onset, flash, t_audio, ruta):
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].imshow(mel_db, aspect="auto", origin="lower",
                 extent=[t_audio[0], t_audio[-1], 0, mel_db.shape[0]])
    ax[0].set_title("1) Energía por banda mel (dB, cruda)")
    ax[1].imshow(norm, aspect="auto", origin="lower", vmin=0, vmax=1,
                 extent=[t_audio[0], t_audio[-1], 0, norm.shape[0]])
    ax[1].set_title("2) Normalizada por banda (percentiles + gamma)")
    ax[2].imshow(suave, aspect="auto", origin="lower", vmin=0, vmax=1,
                 extent=[t_audio[0], t_audio[-1], 0, suave.shape[0]])
    ax[2].set_title("3) Suavizada con ataque/release (lo que controla los tiles)")
    ax[3].plot(t_audio, onset, alpha=0.5, label="onset strength")
    ax[3].plot(t_audio, flash, label="flash suavizado")
    ax[3].set_title("4) Golpes (onsets)")
    ax[3].legend()
    for a in ax[:3]:
        a.set_ylabel("banda")
    ax[3].set_xlabel("tiempo (s)")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close(fig)


def main():
    assert FILAS * COLUMNAS == N_BANDAS, "FILAS * COLUMNAS debe ser igual a N_BANDAS"
    assert 6 <= N_BANDAS <= 12 and FPS >= 10

    y, sr, dur = cargar_audio(RUTA_AUDIO)
    print(f"Audio: {dur:.1f} s -> video de {dur:.1f} s a {FPS} fps")

    mel_db, t_audio = energias_por_banda(y, sr)
    norm = normalizar_bandas(mel_db)
    dt_audio = HOP / sr
    suave = suavizar_ataque_release(norm, dt_audio, T_ATAQUE, T_RELEASE)

    onset = fuerza_de_golpes(y, sr)
    flash = suavizar_ataque_release(onset, dt_audio, T_ATAQUE_ONSET, T_RELEASE_ONSET)

    guardar_figura(mel_db, norm, suave, onset, flash, t_audio, RUTA_FIGURA)

    n_frames = int(dur * FPS)
    t_video = np.arange(n_frames) / FPS
    energias_v = interpolar_a_video(suave, t_audio, t_video)
    flash_v = interpolar_a_video(flash, t_audio, t_video)[0]

    ancho = ANCHO - ANCHO % COLUMNAS
    alto = ALTO - ALTO % FILAS
    img = preparar_imagen(RUTA_IMAGEN, ancho, alto)
    tiles, th, tw = cortar_tiles(img, FILAS, COLUMNAS)

    print("Generando frames...")
    escribir_video(RUTA_VIDEO_MUDO, tiles, th, tw, FILAS, COLUMNAS, energias_v, flash_v, FPS)

    if unir_audio(RUTA_VIDEO_MUDO, RUTA_AUDIO, RUTA_VIDEO_FINAL, INICIO_S, dur):
        print(f"Listo: {RUTA_VIDEO_FINAL}")
    else:
        print(f"Listo (sin audio): {RUTA_VIDEO_MUDO}")
    print(f"Figura para el reporte: {RUTA_FIGURA}")


if __name__ == "__main__":
    main()