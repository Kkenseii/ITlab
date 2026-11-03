# Библиотека фильтров изображений (Python, OpenCV, NumPy)

Практическая работа №1. Фильтры реализованы на базовых матричных операциях NumPy.
OpenCV используется только для чтения/кодирования файлов (`imdecode`, `imencode`).
Высокоуровневые функции (`resize`, `cvtColor`, `filter2D`, `Canny`, …) не применяются.

## Структура

| Файл | Назначение |
|---|---|
| `filters.py` | абстрактный `ImageFilter` (фабрика `get_filter`, метод `apply_filter`) и фильтры-наследники |
| `main.py` | `cli_argument_parser`, `read_image`, сохранение, точка входа |
| `requirements.txt` | зависимости |

## Установка

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```bash
python main.py -f <фильтр> -i <вход> -o <выход> [параметры]
```

Примеры:

```bash
python main.py -f resize  -i photo.jpg -o out.png --width 400 --height 300
python main.py -f resize  -i photo.jpg -o out.png --scale 0.5 --method nearest
python main.py -f gray    -i photo.jpg -o out.png
python main.py -f antique -i photo.jpg -o out.png --intensity 0.9
python main.py -f fade    -i photo.jpg -o out.png --alpha 0.5 --desaturation 0.6
python main.py -f film    -i photo.jpg -o out.png --grain 12 --contrast 1.3 --seed 1
python main.py -f matte   -i photo.jpg -o out.png --radius 0.6 --feather 0.4
python main.py -f aged    -i photo.jpg -o out.png --noise 14 --scratches 50 --dust 400
python main.py -f neon    -i photo.jpg -o out.png --color 0,255,255 --glow_radius 8
```

Ошибки (нет файла, не изображение, неверные параметры) обрабатываются: выводится сообщение
в `stderr`, код возврата 1.

### Фильтры и параметры

| Фильтр (`-f`) | Класс | Параметры (по умолчанию) |
|---|---|---|
| `resize` | `Resize` | `--width`, `--height`, `--scale`, `--method bilinear\|nearest` (bilinear) |
| `gray` | `RGB2GrayScale` | — |
| `antique` | `Antique` | `--intensity` 0..1 (1) |
| `fade` | `FadeColor` | `--alpha` (0.4), `--desaturation` (0.5), `--level` (210) |
| `film` | `FilmEffect` | `--grain` (10), `--contrast` (1.2), `--seed` |
| `matte` | `Matte` | `--radius` (0.65), `--feather` (0.35) |
| `aged` | `AgedPhoto` | `--noise` (12), `--scratches` (40), `--dust` (300), `--seed` |
| `neon` | `Neon` | `--color R,G,B` (255,0,255), `--glow_radius` (6), `--glow_gain` (1.5), `--edge_threshold` (0.1) |

Для `resize` достаточно одного из `--width`/`--height` (пропорции сохраняются) или `--scale`.

## Математическое описание

Изображение — матрица $I(y,x)$ с каналами $B,G,R$; результат всегда приводится операцией
$\mathrm{clip}(\mathrm{round}(v),0,255)$.

### 1. Изменение разрешения (`Resize`)

Исходный размер $W\times H$, целевой $W'\times H'$. Центр пикселя результата $(x',y')$
отображается в исходное изображение:

$$x = \frac{(x'+0.5)\,W}{W'} - 0.5,\qquad y = \frac{(y'+0.5)\,H}{H'} - 0.5,$$

координаты ограничиваются диапазонами $[0,W-1]$, $[0,H-1]$.

**Билинейная интерполяция.** $x_0=\lfloor x\rfloor$, $x_1=\min(x_0+1,W-1)$, $\Delta x=x-x_0$ (аналогично $y$):

$$I'=(1-\Delta y)\big[(1-\Delta x)I_{00}+\Delta x\,I_{01}\big]+\Delta y\big[(1-\Delta x)I_{10}+\Delta x\,I_{11}\big].$$

**Ближайший сосед:** $I'(y',x')=I(\mathrm{round}(y),\mathrm{round}(x))$.

### 2. Оттенки серого (`RGB2GrayScale`)

Взвешенная яркость (ITU-R BT.601):

$$Y = 0.299\,R + 0.587\,G + 0.114\,B.$$

### 3. Антиквариат (`Antique`)

Линейное преобразование цвета в сепию:

$$\begin{pmatrix}R'\\G'\\B'\end{pmatrix}=
\begin{pmatrix}0.393&0.769&0.189\\0.349&0.686&0.168\\0.272&0.534&0.131\end{pmatrix}
\begin{pmatrix}R\\G\\B\end{pmatrix}.$$

Результат: $I_{out}=(1-k)\,I+k\,I'$, где $k$ — `intensity`.

### 4. Выцветание (`FadeColor`)

1. Снижение насыщенности: $C_d = Y + (1-d)(C-Y)$, $d$ — `desaturation`.
2. Подмешивание светлого тона: $C_{out}=(1-\alpha)\,C_d+\alpha\,L$, $\alpha$ — `alpha`, $L$ — `level`.

Эффект: падают насыщенность и контраст, тени поднимаются.

### 5. Имитация плёнки — инфракрасная false-color (`FilmEffect`)

1. Оценка ближнего ИК-сигнала (растительность светлая): $N=\mathrm{clip}(0.2R+0.9G-0.1B,0,255)$.
2. Смещение каналов, как у цветной ИК-плёнки: $R'=N,\; G'=R,\; B'=G$.
3. Контраст: $C'' = c\,(C'-128)+128$.
4. Зерно: $C_{out}=C''+\xi(y,x)$, $\xi\sim\mathcal N(0,\sigma^2)$, одно значение на пиксель для всех каналов.

### 6. Matte — овальная белая рамка (`Matte`)

Нормированный эллиптический радиус ($\rho=0$ в центре, $\rho=1$ на серединах сторон):

$$\rho(y,x)=\sqrt{\Big(\tfrac{y+0.5-H/2}{H/2}\Big)^2+\Big(\tfrac{x+0.5-W/2}{W/2}\Big)^2}.$$

Маска с плавным переходом (smoothstep):

$$t=\mathrm{clip}\Big(\frac{\rho-r}{f},0,1\Big),\qquad m=t^2(3-2t),$$

$$I_{out}=(1-m)\,I+255\,m,$$

где $r$ — `radius`, $f$ — `feather`. Центр не меняется, края становятся белыми.

### 7. Состаренная фотография (`AgedPhoto`)

1. Сепия (см. п. 3), затем выцветший контраст: $C'=0.85\,(C-128)+128+10$.
2. Плёночный шум: $C''=C'+\xi$, $\xi\sim\mathcal N(0,\sigma^2)$.
3. Царапины: каждая — вертикальная линия случайной длины со случайным смещением $x_i=x_0+\lfloor \tfrac12\sum s_j\rfloor$, $s_j\in\{-1,0,0,1\}$. Цвет $v$ — светлый (255, вероятность 0.7) или тёмный (20), прозрачность $a\in[0.3,0.9]$: $C=(1-a)C''+a\,v$.
4. Пыль: в случайных пикселях $C=0.3\,C''+0.7\,v$, $v\in\{255,15\}$.

### 8. Неоновый эффект (`Neon`)

1. Яркость $Y$ (п. 2) со сглаживанием усредняющим окном $3\times3$.
2. Оператор Собеля (свёртка через сдвиги матрицы):

$$G_x=\begin{pmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{pmatrix}*Y,\quad
G_y=\begin{pmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{pmatrix}*Y,\quad
M=\sqrt{G_x^2+G_y^2}.$$

3. Нормировка по 99-му процентилю и порог: $e=\mathrm{clip}\big(\tfrac{M/P_{99}-\tau}{1-\tau},0,1\big)$.
4. Свечение — двукратное усредняющее размытие радиуса $R$ (приближение гауссова): $g=\mathrm{clip}(3\,\kappa\,\mathrm{box}_R(\mathrm{box}_R(e)),0,1)$, $\kappa$ — `glow_gain`. Усреднение по окну $(2R+1)\times(2R+1)$ выполняется раздельно по осям: для $k=2R+1$ складываются $k$ сдвинутых копий матрицы (края дополняются повторением крайних значений) и сумма делится на $k$:
$$\mathrm{box}(y,x)=\frac1k\sum_{i=-R}^{R} A(y+i,x),$$
затем то же по второй оси.
5. Сборка с цветом линий $K$:
$$I_{out}=0.12\,I+\mathrm{clip}(e+0.6\,g,0,1)\,K+0.5\,e^3\,(255-K).$$
Тёмный фон, яркие контуры и светящийся ореол.
