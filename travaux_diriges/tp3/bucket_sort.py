from mpi4py import MPI
import numpy as np
from time import time

def bucket_sort_parallel(N: int, seed: int = 123):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    P = comm.Get_size()

    if rank == 0:
        rng = np.random.default_rng(seed)
        data = rng.random(N)  # floats uniformes dans [0,1)
        buckets = [[] for _ in range(P)]

        for x in data:
            dest = int(x * P)
            if dest == P: 
                dest = P - 1
            buckets[dest].append(float(x))
    else:
        buckets = None

    comm.Barrier()
    t0 = time()

    local = comm.scatter(buckets, root=0)

    local.sort()

    gathered = comm.gather(local, root=0)

    comm.Barrier()
    t1 = time()

    t_max = comm.reduce(t1 - t0, op=MPI.MAX, root=0)

    if rank == 0:
        out = []
        for b in gathered:   
            out.extend(b)
        ok = all(out[i] <= out[i+1] for i in range(len(out)-1))
        return t_max, ok, out
    return None, None, None


if __name__ == "__main__":
    N = 200000
    t, ok, out = bucket_sort_parallel(N)

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print(f"[BucketSort SIMPLE] N={N} P={comm.Get_size()} temps={t:.6f}s tri_ok={ok}")
        print("premiers:", out[:5], "derniers:", out[-5:])

"""
Taille du tableau : N = 200 000
Distribution : uniforme dans [0,1)
Temps mesuré : temps maximal parmi les processus (MPI.MAX)

Tableau des performances :

P (nbp) | Temps T(P) (s) | Speedup S(P) = T(1)/T(P)
---------------------------------------------------
   1    |   0.030753    |   1.000
   2    |   0.023098    |   1.331
   4    |   0.015801    |   1.946
   8    |   0.017895    |   1.719
  16    |   0.014906    |   2.063

Interprétation des résultats — Bucket Sort parallèle (version simple)

Les expériences ont été réalisées pour N = 200 000 éléments et différents
nombres de processus MPI (P = 1, 2, 4, 8, 16).

Les résultats montrent une accélération du temps d’exécution lorsque le nombre
de processus augmente, en particulier entre 1, 2 et 4 processus. Cette amélioration
s’explique par le fait que le tri est effectué en parallèle : chaque processus
est assimilé à un bucket et trie localement une partie des données.

Cependant, le speedup n’est pas linéaire et tend à stagner, voire à diminuer,
lorsque le nombre de processus devient plus grand (par exemple pour P = 8).
Ce comportement est attendu et s’explique par plusieurs facteurs :
- le coût des communications MPI (scatter et gather) devient non négligeable
  devant le temps de calcul local lorsque P augmente ;
- le processus 0 génère les données et effectue le dispatch des valeurs dans
  les buckets, ce qui introduit une surcharge et limite la scalabilité ;
- le temps total mesuré correspond au temps maximal parmi tous les processus,
  de sorte qu’un processus plus lent ou un léger déséquilibre entre buckets
  impacte l’exécution globale ;
- pour une taille de problème relativement modeste, l’overhead lié à MPI et
  à l’exécution Python devient dominant.

Malgré ces limitations, l’algorithme montre un gain réel par rapport à
l’exécution séquentielle, avec un speedup maximal d’environ 2 pour 16 processus.
Ces résultats sont cohérents avec le caractère simple de l’implémentation
(un bucket par processus, intervalles fixes) et avec les principes présentés
en cours.

Une implémentation plus avancée utilisant des intervalles adaptatifs(splitters)
et une redistribution plus fine des données permettrait d’améliorer
l’équilibrage de charge et la scalabilité pour de plus grandes tailles de données.
"""
