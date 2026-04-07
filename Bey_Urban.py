# -*- coding: utf-8 -*-
"""
Simulation LR-FHSS avec modèle radio Beyrouth Urbain
(El Chall, Lahoud, El Helou - IEEE IoT Journal 2019)
+ comparaison h=2, h=3, 50/50, 25/75, 75/25
+ labels dynamiques
+ taux de succès ET goodput (bytes)
+ export Excel
"""
from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings
import simpy
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# Modèle radio Beyrouth Urbain (El Chall et al., 2019)
# PL = 10n × log10(d) + PL0 + Lh × log10(hED)
# ============================================================
N_EXP    = 4.18      # exposant path loss urbain Beyrouth
PL0      = 102.86    # path loss de référence
LH       = -6.3      # facteur hauteur antenne end-device
H_ED     = 1.5       # hauteur antenne end-device (m)
F_MHZ    = 868.0     # fréquence EU (MHz)
P_TX_DBM = 14.0      # puissance max légale ETSI (dBm)

SENSITIVITY = {
    'DR8': -138.0,   # LR-FHSS CR 1/3
    'DR9': -137.0,   # LR-FHSS CR 2/3
}

def path_loss(d_km, h_ed=H_ED):
    return 10 * N_EXP * np.log10(d_km) + PL0 + LH * np.log10(h_ed)

def rssi(d_km, h_ed=H_ED):
    return P_TX_DBM - path_loss(d_km, h_ed)

def is_covered(d_km, dr='DR8', h_ed=H_ED):
    return rssi(d_km, h_ed) >= SENSITIVITY[dr]

def max_distance(dr='DR8', h_ed=H_ED):
    rssi_min = SENSITIVITY[dr]
    log10_d = (P_TX_DBM - PL0 - LH * np.log10(h_ed) - rssi_min) / (10 * N_EXP)
    return 10 ** log10_d

def packet_error_rate(d_km, dr='DR8', h_ed=H_ED):
    margin = rssi(d_km, h_ed) - SENSITIVITY[dr]
    if margin <= 0:
        return 1.0
    elif margin >= 20:
        return 0.0
    else:
        return 1 - (margin / 20)

def make_label(p):
    """Génère un label dynamique selon p."""
    if p == 1.0:
        return 'h=2 fixe'
    elif p == 0.0:
        return 'h=3 fixe'
    else:
        return f'p2={p:.2f}/p3={1-p:.2f}'

# ============================================================
# Fragment étendu avec flag radio_lost
# ============================================================
class FragmentWithRadio(Fragment):
    def __init__(self, type, duration, channel, packet):
        super().__init__(type, duration, channel, packet)
        self.radio_lost = False

# ============================================================
# Packet étendu qui utilise FragmentWithRadio
# ============================================================
class PacketWithRadio(Packet):
    def __init__(self, node_id, obw, headers, payloads,
                 header_duration, payload_duration):
        self.id = id(self)
        self.node_id = node_id
        self.index_transmission = 0
        self.success = 0
        self.channels = random.choices(range(obw), k=headers+payloads)
        self.fragments = []
        for h in range(headers):
            self.fragments.append(
                FragmentWithRadio('header', header_duration,
                                  self.channels[h], self.id))
        for p in range(payloads):
            self.fragments.append(
                FragmentWithRadio('payload', payload_duration,
                                  self.channels[p+h+1], self.id))

# ============================================================
# Node avec prise en compte du PER radio
# ============================================================
class NodeWithRadio(Node):
    def __init__(self, obw, headers, payloads, header_duration,
                 payload_duration, transceiver_wait, traffic_generator,
                 per):
        super().__init__(obw, headers, payloads, header_duration,
                         payload_duration, transceiver_wait,
                         traffic_generator)
        self.per = per
        self.packet = PacketWithRadio(
            self.id, obw, headers, payloads,
            header_duration, payload_duration)

    def end_of_transmission(self):
        self.packet = PacketWithRadio(
            self.id, self.obw, self.headers, self.payloads,
            self.header_duration, self.payload_duration)

    def transmit(self, env, bs):
        while 1:
            yield env.timeout(self.next_transmission())
            self.transmitted += 1
            bs.add_packet(self.packet)
            next_fragment = self.packet.next()
            first_payload = 0
            while next_fragment:
                if first_payload == 0 and next_fragment.type == 'payload':
                    first_payload = 1
                    yield env.timeout(self.transceiver_wait)
                next_fragment.timestamp = env.now

                if random.random() < self.per:
                    yield env.timeout(next_fragment.duration)
                    next_fragment.radio_lost = True
                    next_fragment.transmitted = 1
                    next_fragment.success = 0
                else:
                    bs.check_collision(next_fragment)
                    bs.receive_packet(next_fragment)
                    yield env.timeout(next_fragment.duration)
                    bs.finish_fragment(next_fragment)
                    if self.packet.success == 0:
                        bs.try_decode(self.packet, env.now)

                next_fragment = self.packet.next()
            self.end_of_transmission()

# ============================================================
# BaseACRDA étendu qui tient compte de radio_lost
# ============================================================
class BaseACRDAWithRadio(BaseACRDA):
    def try_decode(self, packet, now):
        for f in list(packet.fragments):
            if not self.in_window(f, now):
                packet.fragments.remove(f)
            else:
                break

        def frag_ok(f):
            radio_lost = getattr(f, 'radio_lost', False)
            return (len(f.collided) == 0) and (f.transmitted == 1) and (not radio_lost)

        h_success = sum(frag_ok(f) if f.type == 'header' else 0 for f in packet.fragments)
        p_success = sum(frag_ok(f) if f.type == 'payload' else 0 for f in packet.fragments)
        success = 1 if ((h_success > 0) and (p_success >= self.threshold)) else 0

        if success == 1:
            self.packets_received[packet.node_id] += 1
            packet.success = 1
            for f in packet.fragments:
                f.success = 1
                for c in list(f.collided):
                    f.collided.remove(c)
                    c.collided.remove(f)
            return True
        else:
            return False

# ============================================================
# Simulation headers fixes → retourne (succès, goodput)
# ============================================================
def run_sim(settings: Settings, distance_km=1.0, dr='DR8',
            h_ed=H_ED, seed=0):
    if not is_covered(distance_km, dr, h_ed):
        return None, None

    per = packet_error_rate(distance_km, dr, h_ed)
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()

    if settings.base == 'acrda':
        bs = BaseACRDAWithRadio(
            settings.obw, settings.window_size, settings.window_step,
            settings.time_on_air, settings.threshold)
        env.process(bs.sic_window(env))
    else:
        bs = Base(settings.obw, settings.threshold)

    nodes = []
    for i in range(settings.number_nodes):
        node = NodeWithRadio(
            settings.obw, settings.headers, settings.payloads,
            settings.header_duration, settings.payload_duration,
            settings.transceiver_wait, settings.traffic_generator,
            per=per)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))

    env.run(until=settings.simulation_time)

    success     = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    if transmitted == 0:
        return 1.0, 0
    return success / transmitted, success * settings.payload_size

# ============================================================
# Simulation distribution mixte → retourne (succès, goodput)
# ============================================================
def run_sim_mixed(number_nodes, distance_km=1.0, dr='DR8',
                  p=0.5, base='acrda', code='1/3',
                  simulation_time=3600, payload_size=10,
                  obw=35, h_ed=H_ED, seed=0):
    if not is_covered(distance_km, dr, h_ed):
        return None, None

    per = packet_error_rate(distance_km, dr, h_ed)
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()

    n2 = int(p * number_nodes)
    n3 = number_nodes - n2

    s2 = Settings(number_nodes=n2, simulation_time=simulation_time,
                  payload_size=payload_size, headers=2, code=code,
                  obw=obw, base=base)
    s3 = Settings(number_nodes=n3, simulation_time=simulation_time,
                  payload_size=payload_size, headers=3, code=code,
                  obw=obw, base=base)

    if base == 'acrda':
        avg_toa = (s2.time_on_air + s3.time_on_air) / 2
        bs = BaseACRDAWithRadio(obw, s2.window_size, s2.window_step,
                                avg_toa, s2.threshold)
        env.process(bs.sic_window(env))
    else:
        bs = Base(obw, s2.threshold)

    nodes = []
    for _ in range(n2):
        node = NodeWithRadio(s2.obw, s2.headers, s2.payloads,
                             s2.header_duration, s2.payload_duration,
                             s2.transceiver_wait, s2.traffic_generator,
                             per=per)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))

    for _ in range(n3):
        node = NodeWithRadio(s3.obw, s3.headers, s3.payloads,
                             s3.header_duration, s3.payload_duration,
                             s3.transceiver_wait, s3.traffic_generator,
                             per=per)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))

    env.run(until=simulation_time)

    success     = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    if transmitted == 0:
        return 1.0, 0
    return success / transmitted, success * payload_size


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # =========================================================
    # ← Modifiez ces paramètres
    DR      = 'DR8'
    N_NODES = 150000 // 8
    BASE    = 'acrda'
    CODE    = '1/3'
    SEED    = 0
    H_ED_M  = 1.5    # hauteur end-device (m) : 0.2, 1.5, 3.0

    distances = [1,2,3,4,5,6,8,10,12,14]

    # ← Ajoutez / retirez des proportions ici librement
    # 1.0 = 100% h=2, 0.0 = 100% h=3, autre = mixte
    PROPORTIONS = [1.0, 0.0, 0.5, 0.75, 0.25]
    # =========================================================

    # Labels générés automatiquement
    DISTRIBUTIONS = [(p, make_label(p)) for p in PROPORTIONS]

    # Settings pour h=2 et h=3 fixes
    s2 = Settings(number_nodes=N_NODES, simulation_time=3600,
                  payload_size=10, headers=2, code=CODE,
                  obw=35, base=BASE)
    s3 = Settings(number_nodes=N_NODES, simulation_time=3600,
                  payload_size=10, headers=3, code=CODE,
                  obw=35, base=BASE)

    print("=== Modèle : Beyrouth Urbain (El Chall et al., 2019) ===")
    print(f"  Fréquence  : {F_MHZ} MHz")
    print(f"  P_tx       : {P_TX_DBM} dBm")
    print(f"  n (exposant): {N_EXP}")
    print(f"  PL0        : {PL0} dB")
    print(f"  h_ED       : {H_ED_M} m")
    print()
    print("=== Distances maximales de couverture ===")
    for dr_name in SENSITIVITY:
        print(f"  {dr_name} : {max_distance(dr_name, H_ED_M):.2f} km")
    print()
    print("=== Impact hauteur end-device sur portée DR8 ===")
    for h in [0.2, 1.5, 2.0, 3.0]:
        print(f"  h_ED={h}m : {max_distance('DR8', h):.2f} km")
    print()
    print(f"=== Résultats ({DR}, {BASE}, {N_NODES}×8={N_NODES*8} noeuds, seed={SEED}) ===")
    print(f"  h_ED={H_ED_M}m | code={CODE}")
    print()

    rows = []

    for d in distances:
        per_val  = packet_error_rate(d, DR, H_ED_M)
        rssi_val = rssi(d, H_ED_M)

        print(f"--- Distance={d} km | RSSI={rssi_val:.1f} dBm | PER={per_val*100:.1f}% ---")

        row = {
            'Distance (km)': d,
            'RSSI (dBm)'   : round(rssi_val, 1),
            'PER (%)'      : round(per_val * 100, 1),
        }

        if not is_covered(d, DR, H_ED_M):
            print(f"  ⚠️  Hors portée pour toutes les configurations")
            for p, label in DISTRIBUTIONS:
                row[f'{label} Succès (%)']  = None
                row[f'{label} Goodput (B)'] = None
            rows.append(row)
            print()
            continue

        for p, label in DISTRIBUTIONS:
            if p == 1.0:
                r, g = run_sim(s2, distance_km=d, dr=DR,
                               h_ed=H_ED_M, seed=SEED)
            elif p == 0.0:
                r, g = run_sim(s3, distance_km=d, dr=DR,
                               h_ed=H_ED_M, seed=SEED)
            else:
                r, g = run_sim_mixed(N_NODES, distance_km=d, dr=DR,
                                     p=p, base=BASE, code=CODE,
                                     h_ed=H_ED_M, seed=SEED)

            if r is None:
                print(f"  {label:<20} → hors portée")
                row[f'{label} Succès (%)']  = None
                row[f'{label} Goodput (B)'] = None
            else:
                print(f"  {label:<20} → succès={r*100:.2f}%  goodput={g} B")
                row[f'{label} Succès (%)']  = round(r * 100, 2)
                row[f'{label} Goodput (B)'] = g

        rows.append(row)
        print()
'''

    # ============================================================
    # Export Excel
    # ============================================================
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"resultats_beyrouth_urbain_{DR}_{N_NODES*8}noeuds_{timestamp}.xlsx"

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Résultats', index=False)

        params_data = {
            'Paramètre': [
                'Modèle radio', 'Fréquence (MHz)', 'P_tx (dBm)',
                'n (exposant)', 'PL0 (dB)', 'Lh',
                'h_ED (m)', 'DR', 'Nœuds (sim)',
                'Nœuds (réseau)', 'Base', 'Code rate',
                'Seed', 'Portée DR8 (km)', 'Distributions testées'
            ],
            'Valeur': [
                'Beyrouth Urbain (El Chall et al., 2019)',
                F_MHZ, P_TX_DBM,
                N_EXP, PL0, LH,
                H_ED_M, DR, N_NODES,
                N_NODES * 8, BASE, CODE,
                SEED, round(max_distance('DR8', H_ED_M), 2),
                str([label for _, label in DISTRIBUTIONS])
            ]
        }
        pd.DataFrame(params_data).to_excel(
            writer, sheet_name='Paramètres', index=False)

    print(f"✅ Résultats exportés → {filename}")
'''