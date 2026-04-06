# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:33:07 2026

@author: Tanios
"""

# -*- coding: utf-8 -*-
"""
Simulation LR-FHSS avec distribution mixte de headers :
  - 50% des noeuds transmettent avec 2 headers
  - 50% des noeuds transmettent avec 3 headers
N = 17000 noeuds, code rate 1/3, base core et acrda
"""
from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings
import simpy
import numpy as np


def run_sim_mixed(base='acrda', number_nodes=150000//8, seed=0,
                  simulation_time=3600, payload_size=10,
                  code='1/3', obw=35):
    """
    Lance une simulation avec 50% des noeuds à 2 headers
    et 50% à 3 headers.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()

    # Settings pour le groupe 2 headers
    s2 = Settings(
        number_nodes    = number_nodes // 2,
        simulation_time = simulation_time,
        payload_size    = payload_size,
        headers         = 2,
        code            = code,
        obw             = obw,
        base            = base
    )

    # Settings pour le groupe 3 headers
    s3 = Settings(
        number_nodes    = number_nodes // 2,
        simulation_time = simulation_time,
        payload_size    = payload_size,
        headers         = 3,
        code            = code,
        obw             = obw,
        base            = base
    )

    # Créer la gateway (une seule, partagée par tous les noeuds)
    if base == 'acrda':
        # time_on_air moyen entre les deux groupes
        avg_toa = (s2.time_on_air + s3.time_on_air) / 2
        bs = BaseACRDA(obw, s2.window_size, s2.window_step, avg_toa, s2.threshold)
        env.process(bs.sic_window(env))
    else:
        bs = Base(obw, s2.threshold)

    nodes = []

    # Groupe 1 : 50% avec 2 headers
    for i in range(number_nodes // 2):
        node = Node(
            s2.obw, s2.headers, s2.payloads,
            s2.header_duration, s2.payload_duration,
            s2.transceiver_wait, s2.traffic_generator
        )
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))

    # Groupe 2 : 50% avec 3 headers
    for i in range(number_nodes // 2):
        node = Node(
            s3.obw, s3.headers, s3.payloads,
            s3.header_duration, s3.payload_duration,
            s3.transceiver_wait, s3.traffic_generator
        )
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))

    # Lancer la simulation
    env.run(until=simulation_time)

    success     = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    return success / transmitted if transmitted > 0 else 1.0


if __name__ == "__main__":

    NUMBER_NODES    = 25000//8
    SIMULATION_TIME = 3600
    CODE            = '1/3'

    print(f"N={NUMBER_NODES} | code={CODE} | sim_time=3600 sec")
    print(f"Distribution : 50% × 2 headers + 50% × 3 headers")
    print("-" * 55)

    for base in ['core', 'acrda']:
        result = run_sim_mixed(
            base            = base,
            number_nodes    = NUMBER_NODES,
            simulation_time = SIMULATION_TIME,
            code            = CODE
        )
        print(f"base={base:<5} | Prob succès = {result:.6f}")

    print("\n--- Comparaison avec headers fixes ---")
    # Headers=2 fixe
    for base in ['core', 'acrda']:
        s = Settings(number_nodes=NUMBER_NODES, simulation_time=SIMULATION_TIME,
                     payload_size=10, headers=2, code=CODE, base=base)
        from Tanios_Run_Two import run_sim
        prob, _, _ = run_sim(s)
        print(f"base={base:<5} | headers=2 fixe  | Prob succès = {prob:.6f}")

    # Headers=3 fixe
    for base in ['core', 'acrda']:
        s = Settings(number_nodes=NUMBER_NODES, simulation_time=SIMULATION_TIME,
                     payload_size=10, headers=3, code=CODE, base=base)
        prob, _, _ = run_sim(s)
        print(f"base={base:<5} | headers=3 fixe  | Prob succès = {prob:.6f}")