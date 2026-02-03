import numpy as np
from dataclasses import dataclass
from math import log
import matplotlib.cm
from PIL import Image
from mpi4py import MPI


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth: bool = False, clamp: bool = True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        if not clamp:
            return value
        return max(0.0, min(value, 1.0))

    def count_iterations(self, c: complex, smooth: bool = False) -> float:
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return float(self.max_iterations)
        if (c.real + 1.0) * (c.real + 1.0) + c.imag * c.imag < 0.0625:
            return float(self.max_iterations)
        if (-0.75 < c.real < 0.5):
            ct = (c.real - 0.25) + 1j * c.imag
            r = abs(ct)
            if r < 0.5 * (1.0 - ct.real / max(r, 1e-14)):
                return float(self.max_iterations)

        z = 0j
        for it in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return it + 1.0 - log(log(abs(z))) / log(2.0)
                return float(it)
        return float(self.max_iterations)


def row_block(rank: int, size: int, height: int) -> tuple[int, int]:
    base = height // size
    rem = height % size
    y0 = rank * base + min(rank, rem)
    nloc = base + (1 if rank < rem else 0)
    return y0, y0 + nloc


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10.0)
width, height = 1024, 1024

scale_x = 3.0 / width
scale_y = 2.25 / height


def compute_row(y: int) -> np.ndarray:
    xs = np.arange(width, dtype=np.int64)
    c = (-2.0 + scale_x * xs) + 1j * (-1.125 + scale_y * y)
    out = np.empty((width,), dtype=np.double)
    for x in range(width):
        out[x] = mandelbrot_set.convergence(complex(c[x]), smooth=True)
    return out


def q1_block_gatherv() -> float:
    y0, y1 = row_block(rank, size, height)
    nloc = y1 - y0

    conv_loc = np.empty((nloc, width), dtype=np.double)

    comm.Barrier()
    t0 = MPI.Wtime()

    for j, y in enumerate(range(y0, y1)):
        conv_loc[j, :] = compute_row(y)

    t1 = MPI.Wtime()
    t_max = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

    sendbuf = conv_loc.ravel()
    counts = np.array(
        [(row_block(r, size, height)[1] - row_block(r, size, height)[0]) * width for r in range(size)],
        dtype=np.int64,
    )
    displs = np.zeros(size, dtype=np.int64)
    displs[1:] = np.cumsum(counts[:-1])

    if rank == 0:
        full = np.empty((height * width,), dtype=np.double)
    else:
        full = None

    comm.Gatherv(sendbuf, [full, counts, displs, MPI.DOUBLE], root=0)

    if rank == 0:
        convergence = full.reshape((height, width))
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        image.save("mandelbrot_p1_q1.png")

    return t_max if rank == 0 else 0.0


def q2_cyclic_static() -> float:
    ys = list(range(rank, height, size))
    nloc = len(ys)

    conv_loc = np.empty((nloc, width), dtype=np.double)

    comm.Barrier()
    t0 = MPI.Wtime()

    for j, y in enumerate(ys):
        conv_loc[j, :] = compute_row(y)

    t1 = MPI.Wtime()
    t_max = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

    counts = comm.gather(nloc, root=0)

    if rank == 0:
        convergence = np.empty((height, width), dtype=np.double)
        y_arr = np.array(ys, dtype=np.int32)
        convergence[y_arr, :] = conv_loc

        for src in range(1, size):
            nsrc = counts[src]
            y_src = np.empty((nsrc,), dtype=np.int32)
            pix_src = np.empty((nsrc, width), dtype=np.double)
            comm.Recv(y_src, source=src, tag=10)
            comm.Recv(pix_src, source=src, tag=11)
            convergence[y_src, :] = pix_src

        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        image.save("mandelbrot_p1_q2.png")

        return t_max
    else:
        comm.Send(np.array(ys, dtype=np.int32), dest=0, tag=10)
        comm.Send(conv_loc, dest=0, tag=11)
        return 0.0


TAG_WORK = 1
TAG_STOP = 2
TAG_RESULT = 3


def q3_master_worker() -> float:
    comm.Barrier()
    t0 = MPI.Wtime()

    if rank == 0:
        convergence = np.empty((height, width), dtype=np.double)

        next_y = 0
        done = 0

        for dst in range(1, size):
            if next_y < height:
                comm.send(next_y, dest=dst, tag=TAG_WORK)
                next_y += 1
            else:
                comm.send(None, dest=dst, tag=TAG_STOP)

        status = MPI.Status()
        while done < height:
            y, row = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
            src = status.Get_source()

            convergence[y, :] = row
            done += 1

            if next_y < height:
                comm.send(next_y, dest=src, tag=TAG_WORK)
                next_y += 1
            else:
                comm.send(None, dest=src, tag=TAG_STOP)

        comm.Barrier()
        t1 = MPI.Wtime()

        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
        image.save("mandelbrot_p1_q3.png")

        return t1 - t0
    else:
        status = MPI.Status()
        while True:
            y = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == TAG_STOP:
                break
            comm.send((y, compute_row(y)), dest=0, tag=TAG_RESULT)

        comm.Barrier()
        return 0.0


t_q1 = q1_block_gatherv()
if rank == 0:
    print(f"[Q1] nbp={size} temps={t_q1:.6f}s")

t_q2 = q2_cyclic_static()
if rank == 0:
    print(f"[Q2] nbp={size} temps={t_q2:.6f}s")

t_q3 = q3_master_worker()
if rank == 0:
    print(f"[Q3] nbp={size} temps={t_q3:.6f}s")

# Réponses (résultats et interpretation)

# Question 1 – Répartition par blocs contigus
#
# Taille de l’image : 1024 × 1024
#
# Temps mesurés (en secondes)
#
# p (nbp) | Temps T(p)
# --------------------
#   1     |   7.02
#   4     |   2.28
#   8     |   1.31
#  16     |   0.78
#
# Speedup (référence T(1))
#
# p (nbp) | Speedup
# -----------------
#   1     |  1.00
#   4     |  3.08
#   8     |  5.36
#  16     |  9.00
#
# Analyse :
# La parallélisation permet de réduire nettement le temps de calcul,
# mais le speedup reste inférieur au cas idéal. Certaines lignes de
# l’image sont plus coûteuses à calculer que d’autres, ce qui entraîne
# un déséquilibre de charge entre les processus.
#
# De plus, la reconstruction finale de l’image sur le processus 0
# reste une partie séquentielle qui limite les performances globales.


# Question 2 – Répartition cyclique statique
#
# Temps mesurés (en secondes)
#
# p (nbp) | Blocs contigus | Cyclique
# ----------------------------------
#   1     |     6.95      |   6.80
#   4     |     2.31      |   1.84
#   8     |     1.29      |   0.93
#  16     |     0.79      |   0.55
#
# Speedup (référence T(1))
#
# p (nbp) | S_blocs | S_cyclique
# ------------------------------
#   1     |  1.00   |   1.00
#   4     |  3.01   |   3.70
#   8     |  5.39   |   7.31
#  16     |  8.80   |  12.36
#
# Analyse :
# La répartition cyclique améliore clairement les performances par
# rapport aux blocs contigus. Les lignes coûteuses sont mieux réparties
# entre les processus, ce qui réduit les temps d’attente et améliore
# l’équilibrage de charge.
#
# Le coût de communication est légèrement plus élevé, mais il est
# largement compensé par le gain sur le calcul.


# Question 3 – Stratégie maître-esclave
#
# Temps mesurés (en secondes)
#
# p (nbp) | Q1 blocs | Q2 cyclique | Q3 maître-esclave
# ---------------------------------------------------
#   4     |  2.26    |   1.83      |   2.21
#   8     |  1.30    |   0.92      |   0.95
#  16     |  0.80    |   0.56      |   0.52
#  32     |  0.74    |   0.50      |   0.47
#
# Speedup (référence T(1) ≈ 6.9 s)
#
# p (nbp) | S_blocs | S_cyclique | S_maître-esclave
# -------------------------------------------------
#   4     |  3.05   |   3.77     |      3.12
#   8     |  5.31   |   7.50     |      7.26
#  16     |  8.63   |  12.32     |     13.27
#  32     |  9.32   |  13.80     |     14.68
#
# Analyse :
# La stratégie maître-esclave permet un équilibrage dynamique de la
# charge, ce qui évite qu’un processus reste inactif. Les performances
# sont comparables à la répartition cyclique pour un nombre modéré de
# processus, et deviennent légèrement meilleures lorsque p augmente.
#
# En revanche, le processus maître peut devenir un point de congestion,
# car il centralise toutes les communications. Cette approche reste
# donc efficace, mais sa scalabilité est limitée à très grande échelle.

