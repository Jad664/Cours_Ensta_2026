import numpy as np
from mpi4py import MPI

dim = 120

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if dim % size != 0:
    if rank == 0:
        print("Erreur : dim doit être divisible par le nombre de processus (nbp).")
    raise SystemExit(1)

nloc = dim // size

u = np.arange(1.0, dim + 1.0, dtype=np.double)


def matvec_by_columns() -> float:
    j0 = rank * nloc
    j1 = (rank + 1) * nloc

    i_idx = np.arange(dim, dtype=np.int64)[:, None]
    j_idx = np.arange(j0, j1, dtype=np.int64)[None, :]
    a_block = ((i_idx + j_idx) % dim).astype(np.double) + 1.0

    comm.Barrier()
    t0 = MPI.Wtime()

    v_part = a_block @ u[j0:j1]
    v = np.empty((dim,), dtype=np.double)
    comm.Allreduce(v_part, v, op=MPI.SUM)

    t1 = MPI.Wtime()
    return comm.reduce(t1 - t0, op=MPI.MAX, root=0)


def matvec_by_rows() -> float:
    i0 = rank * nloc
    i1 = (rank + 1) * nloc

    i_idx = np.arange(i0, i1, dtype=np.int64)[:, None]
    j_idx = np.arange(dim, dtype=np.int64)[None, :]
    a_rows = ((i_idx + j_idx) % dim).astype(np.double) + 1.0

    comm.Barrier()
    t0 = MPI.Wtime()

    v_loc = a_rows @ u
    v = np.empty((dim,), dtype=np.double)
    comm.Allgather([v_loc, MPI.DOUBLE], [v, MPI.DOUBLE])

    t1 = MPI.Wtime()
    return comm.reduce(t1 - t0, op=MPI.MAX, root=0)


t_col = matvec_by_columns()
if rank == 0:
    print(f"[Q1 colonnes] dim={dim} nbp={size} nloc={nloc} temps={t_col:.6f}s")

t_row = matvec_by_rows()
if rank == 0:
    print(f"[Q2 lignes]   dim={dim} nbp={size} nloc={nloc} temps={t_row:.6f}s")

# ============================================================
# Mesures expérimentales – Produit matrice-vecteur MPI
#
# Dimension du problème : N = 120
#
# Temps mesurés (en secondes)
#
# p (nbp) | Q1 – découpage par colonnes | Q2 – découpage par lignes
# ---------------------------------------------------------------
#   1     |           0.0091           |           0.0076
#   2     |           0.0058           |           0.0049
#   3     |           0.0045           |           0.0038
#   4     |           0.0029           |           0.0028
#   8     |           0.0021           |           0.0015
#  12     |           0.0023           |           0.0019
#
# Speedup (référence = temps à p=1 pour chaque méthode)
#
# p (nbp) | S_Q1 colonnes | S_Q2 lignes
# ------------------------------------
#   1     |     1.00      |     1.00
#   2     |     1.57      |     1.55
#   3     |     2.02      |     2.00
#   4     |     3.14      |     2.71
#   8     |     4.33      |     5.07
#  12     |     3.96      |     4.00
#
# Analyse :
# Les deux approches montrent une accélération correcte quand le
# nombre de processus augmente, mais le speedup n’est pas parfaitement
# régulier. Le problème est de petite taille, donc le temps de calcul
# devient rapidement comparable au coût des communications MPI.
#
# Le découpage par colonnes nécessite une réduction globale sur tout
# le vecteur résultat, ce qui peut devenir coûteux lorsque p augmente.
# Le découpage par lignes évite cette réduction et est souvent plus
# efficace pour un nombre intermédiaire de processus.
#
# Dans tous les cas, les performances sont limitées par l’overhead MPI
# et le faible volume de calcul.
