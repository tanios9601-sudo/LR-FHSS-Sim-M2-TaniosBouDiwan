# -*- coding: utf-8 -*-
"""
Simulation LR-FHSS avec distribution mixte :
  - 50% des noeuds avec h=4
  - 50% des noeuds avec h=5
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
SEEDS           = list(range(5))
SIMULATION_TIME = 3600
PAYLOAD_SIZE    = 10
CODE            = '1/3'
OBW             = 35
BASE            = 'acrda'




# ============================================================
# Simulation mixte h4=0.5 / h5=0.5
# ============================================================
def run_sim_mixed(number_nodes, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()

    n4 = number_nodes // 2
    n5 = number_nodes - n4  # prend le reste pour éviter les arrondis

    s4 = Settings(number_nodes=n4, simulation_time=SIMULATION_TIME,
                  payload_size=PAYLOAD_SIZE, headers=4, code=CODE,
                  obw=OBW, base=BASE)
    s5 = Settings(number_nodes=n5, simulation_time=SIMULATION_TIME,
                  payload_size=PAYLOAD_SIZE, headers=5, code=CODE,
                  obw=OBW, base=BASE)

    avg_toa = (s4.time_on_air + s5.time_on_air) / 2
    bs = BaseACRDA(OBW, s4.window_size, s4.window_step, avg_toa, s4.threshold)
    env.process(bs.sic_window(env))

    nodes = []

    for _ in range(n4):
        node = Node(s4.obw, s4.headers, s4.payloads,
                    s4.header_duration, s4.payload_duration,
                    s4.transceiver_wait, s4.traffic_generator)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))

    for _ in range(n5):
        node = Node(s5.obw, s5.headers, s5.payloads,
                    s5.header_duration, s5.payload_duration,
                    s5.transceiver_wait, s5.traffic_generator)
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
def job(n_nodes, seed):
    rate = run_sim_mixed(number_nodes=n_nodes // 8, seed=seed)
    return n_nodes, seed, rate

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    jobs = [(n, s) for n in NODE_COUNTS for s in SEEDS]

    print(f"Distribution : 50% h=4 + 50% h=5")
    print(f"Total jobs : {len(jobs)}  ({len(NODE_COUNTS)} charges x {len(SEEDS)} seeds)")
    print("Lancement en parallele...\n")

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(job)(n, s) for n, s in jobs
    )

    bucket = defaultdict(list)
    for n, s, rate in results:
        bucket[n].append(rate)

    print(f"{'Noeuds reseau':<16} {'Noeuds sim':<12} {'Succes moy (%)':<18} {'Std (%)'}")
    print("-" * 55)

    for n in NODE_COUNTS:
        rates = bucket[n]
        mean  = np.mean(rates) * 100
        std   = np.std(rates)  * 100
        print(f"  {n:<14} {n//8:<12} {mean:<18.2f} {std:.2f}")

    elapsed = time() - start_time
    print(f"\nElapsed time: {elapsed:.2f} seconds")