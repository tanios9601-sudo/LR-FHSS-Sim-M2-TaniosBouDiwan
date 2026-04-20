# -*- coding: utf-8 -*-
"""
Simulation LR-FHSS avec distributions mixtes :
  - h4=0.75 / h5=0.25
  - h4=0.25 / h5=0.75
  - h2=h3=h4=h5=0.25
Noeuds : 25k, 50k, 75k, 100k, 125k, 150k
Moyenne sur 5 seeds (0-4), parallélisation joblib
"""
from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings

import simpy
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from time import time

start_time = time()

# ============================================================
# Paramètres
# ============================================================
NODE_COUNTS     = [25000, 50000, 75000, 100000, 125000, 150000]
SEEDS           = list(range(10))
SIMULATION_TIME = 3600
PAYLOAD_SIZE    = 10
CODE            = '1/3'
OBW             = 35
BASE            = 'acrda'

# Distributions : liste de (label, [(headers, proportion), ...])
DISTRIBUTIONS = [
    #('h4=0.75/h5=0.25', [(4, 0.75), (5, 0.25)]),
    #('h4=0.25/h5=0.75', [(4, 0.25), (5, 0.75)]),
    #('h2=h3=h4=h5=0.25', [(2, 0.25), (3, 0.25), (4, 0.25), (5, 0.25)]),
    ('h4=0.5/h5=0.5', [(4, 0.5), (5, 0.5)])
]

# ============================================================
# Simulation générique multi-groupes
# ============================================================
def run_sim_groups(groups_spec, total_nodes, seed=0):
    """
    groups_spec : [(headers, proportion), ...]
    total_nodes : nombre de noeuds simulés (déjà divisé par 8)
    """
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()

    # Créer un Settings par groupe
    settings_list = []
    for h, prop in groups_spec:
        s = Settings(number_nodes=total_nodes, simulation_time=SIMULATION_TIME,
                     payload_size=PAYLOAD_SIZE, headers=h, code=CODE,
                     obw=OBW, base=BASE)
        settings_list.append((s, prop))

    avg_toa = np.mean([s.time_on_air for s, _ in settings_list])
    s0 = settings_list[0][0]
    bs = BaseACRDA(OBW, s0.window_size, s0.window_step, avg_toa, s0.threshold)
    env.process(bs.sic_window(env))

    nodes = []
    remaining = total_nodes
    for i, (s, prop) in enumerate(settings_list):
        # dernier groupe prend le reste pour éviter les erreurs d'arrondi
        n = int(prop * total_nodes) if i < len(settings_list) - 1 else remaining
        remaining -= n
        for _ in range(n):
            node = Node(s.obw, s.headers, s.payloads,
                        s.header_duration, s.payload_duration,
                        s.transceiver_wait, s.traffic_generator)
            bs.add_node(node.id)
            nodes.append(node)
            env.process(node.transmit(env, bs))

    env.run(until=SIMULATION_TIME)

    success     = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    return success / transmitted if transmitted > 0 else 1.0

# ============================================================
# Job unitaire pour joblib
# ============================================================
def job(label, groups_spec, n_nodes, seed):
    rate = run_sim_groups(groups_spec, total_nodes=n_nodes // 8, seed=seed)
    return label, n_nodes, rate

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    jobs = [
        (label, groups_spec, n, s)
        for label, groups_spec in DISTRIBUTIONS
        for n in NODE_COUNTS
        for s in SEEDS
    ]

    print(f"Total jobs : {len(jobs)}  "
          f"({len(DISTRIBUTIONS)} distributions x {len(NODE_COUNTS)} charges x {len(SEEDS)} seeds)")
    print("Lancement en parallele...\n")

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(job)(label, groups_spec, n, s)
        for label, groups_spec, n, s in jobs
    )

    bucket = defaultdict(list)
    for label, n, rate in results:
        bucket[(label, n)].append(rate)

    for label, groups_spec in DISTRIBUTIONS:
        print(f"=== Distribution : {label} ===")
        print(f"  {'Noeuds reseau':<16} {'Noeuds sim':<12} {'Succes moy (%)':<18} {'Std (%)'}")
        print("  " + "-" * 53)
        for n in NODE_COUNTS:
            rates = bucket[(label, n)]
            mean  = np.mean(rates) * 100
            std   = np.std(rates)  * 100
            print(f"  {n:<16} {n//8:<12} {mean:<18.2f} {std:.2f}")
        print()

    elapsed = time() - start_time
    print(f"Elapsed time: {elapsed:.2f} seconds")