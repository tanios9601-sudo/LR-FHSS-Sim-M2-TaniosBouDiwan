# -*- coding: utf-8 -*-
"""
Run LR-FHSS pour h=4 et h=5
Noeuds : 25k, 50k, 75k, 100k, 125k, 150k
Moyenne sur 5 seeds (0-4)
Parallélisation joblib
"""
from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings

import simpy
import numpy as np
from joblib import Parallel, delayed
from time import time

start_time = time()

# ============================================================
# Paramètres
# ============================================================
HEADERS         = [2,4, 5]
NODE_COUNTS     = [25000, 50000, 75000, 100000, 125000, 150000]
SEEDS           = list(range(1))   # 0, 1, 2, 3, 4
SIMULATION_TIME = 3600
PAYLOAD_SIZE    = 10
CODE            = '1/3'
OBW             = 35
BASE            = 'acrda'

# ============================================================
# Fonction de simulation (1 run = 1 seed)
# ============================================================
def run_sim(settings: Settings, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()

    if settings.base == 'acrda':
        bs = BaseACRDA(settings.obw, settings.window_size, settings.window_step,
                       settings.time_on_air, settings.threshold)
        env.process(bs.sic_window(env))
    else:
        bs = Base(settings.obw, settings.threshold)

    nodes = []
    for _ in range(settings.number_nodes):
        node = Node(settings.obw, settings.headers, settings.payloads,
                    settings.header_duration, settings.payload_duration,
                    settings.transceiver_wait, settings.traffic_generator)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))

    env.run(until=settings.simulation_time)

    success     = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    if transmitted == 0:
        return 1.0
    return success / transmitted

# ============================================================
# Job unitaire pour joblib
# ============================================================
def job(headers, n_nodes, seed):
    s = Settings(
        number_nodes    = n_nodes // 8,
        simulation_time = SIMULATION_TIME,
        payload_size    = PAYLOAD_SIZE,
        headers         = headers,
        code            = CODE,
        obw             = OBW,
        base            = BASE,
    )
    return headers, n_nodes, seed, run_sim(s, seed)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # Construction de tous les jobs
    jobs = [
        (h, n, seed)
        for h    in HEADERS
        for n    in NODE_COUNTS
        for seed in SEEDS
    ]

    print(f"Total jobs : {len(jobs)}  ({len(HEADERS)} headers × "
          f"{len(NODE_COUNTS)} charges × {len(SEEDS)} seeds)")
    print("Lancement en parallèle...\n")

    # Exécution parallèle
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(job)(h, n, s) for h, n, s in jobs
    )

    # --------------------------------------------------------
    # Agrégation : moyenne + std par (headers, n_nodes)
    # --------------------------------------------------------
    from collections import defaultdict
    bucket = defaultdict(list)
    for h, n, seed, rate in results:
        bucket[(h, n)].append(rate)

    print(f"{'Headers':<10} {'Noeuds réseau':<16} {'Noeuds sim':<12} "
          f"{'Succès moy (%)':<18} {'Std (%)':<10} ")
    print("-" * 75)

    for h in HEADERS:
        for n in NODE_COUNTS:
            rates = bucket[(h, n)]
            mean  = np.mean(rates) * 100
            std   = np.std(rates)  * 100
            print(f"  h={h:<7} {n:<16} {n//8:<12} "
                  f"{mean:<18.2f} {std:<10.2f} ")
        print()

    elapsed = time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")