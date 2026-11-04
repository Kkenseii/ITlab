import argparse
import sys

import cv2
import numpy as np

from filters import ImageFilter, FILTERS


def parse_color(text: str):
    try:
        parts = tuple(int(p) for p in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("цвет должен быть в формате R,G,B")
    if len(parts) != 3 or any(not 0 <= p <= 255 for p in parts):
        raise argparse.ArgumentTypeError("цвет: три числа 0..255 через запятую")
    return parts


def cli_argument_parser(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Применение фильтра к изображению (базовые операции над матрицами).")
    parser.add_argument("-f", "--filter", required=True,
                        choices=sorted(FILTERS),
                        help="тип фильтра")
    parser.add_argument("-i", "--input", required=True, help="путь к входному изображению")
    parser.add_argument("-o", "--output", default="result.png", help="путь к результату")

    g = parser.add_argument_group("resize")
    g.add_argument("--width", type=int, help="новая ширина, px")
    g.add_argument("--height", type=int, help="новая высота, px")
    g.add_argument("--scale", type=float, help="коэффициент масштабирования")
    g.add_argument("--method", choices=["bilinear", "nearest"], help="интерполяция")

    g = parser.add_argument_group("antique")
    g.add_argument("--intensity", type=float, help="сила сепии, 0..1 (1)")

    g = parser.add_argument_group("fade")
    g.add_argument("--alpha", type=float, help="доля светлого тона, 0..1 (0.4)")
    g.add_argument("--desaturation", type=float, help="снижение насыщенности, 0..1 (0.5)")
    g.add_argument("--level", type=float, help="яркость тона выцветания, 0..255 (210)")

    g = parser.add_argument_group("film")
    g.add_argument("--grain", type=float, help="СКО зерна (10)")
    g.add_argument("--contrast", type=float, help="коэффициент контраста (1.2)")

    g = parser.add_argument_group("matte")
    g.add_argument("--radius", type=float, help="радиус чистой зоны, 0..1.4 (0.65)")
    g.add_argument("--feather", type=float, help="ширина перехода к белому (0.35)")

    g = parser.add_argument_group("aged")
    g.add_argument("--noise", type=float, help="СКО шума (12)")
    g.add_argument("--scratches", type=int, help="число царапин (40)")
    g.add_argument("--dust", type=int, help="число пылинок (300)")

    g = parser.add_argument_group("neon")
    g.add_argument("--color", type=parse_color, help="цвет линий R,G,B (255,0,255)")
    g.add_argument("--glow_radius", type=int, help="радиус свечения, px (6)")
    g.add_argument("--glow_gain", type=float, help="усиление свечения (1.5)")
    g.add_argument("--edge_threshold", type=float, help="порог контуров, 0..1 (0.1)")

    parser.add_argument("--seed", type=int, help="seed генератора случайных чисел (film, aged)")
    return parser.parse_args(argv)


def read_image(path: str) -> np.ndarray:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as e:
        raise FileNotFoundError(f"Не удалось открыть файл '{path}': {e}")
    if data.size == 0:
        raise ValueError(f"Файл '{path}' пуст")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Файл '{path}' не является поддерживаемым изображением")
    return image


def save_image(path: str, image: np.ndarray) -> None:
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение в формат '{ext}'")
    try:
        buf.tofile(path)
    except OSError as e:
        raise OSError(f"Не удалось сохранить '{path}': {e}")


def main(argv=None) -> int:
    args = cli_argument_parser(argv)
    try:
        image = read_image(args.input)
        params = {k: v for k, v in vars(args).items() if k not in ("filter", "input", "output")}
        flt = ImageFilter.get_filter(args.filter, **params)
        result = flt.apply_filter(image)
        save_image(args.output, result)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print(f"Готово: {args.output} ({result.shape[1]}x{result.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())