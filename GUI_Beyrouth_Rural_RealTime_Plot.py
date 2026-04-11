# -*- coding: utf-8 -*-
"""
Simulation LR-FHSS - Beyrouth Rural (Bekaa Valley)
+ Interface graphique tkinter pour configurer les paramètres
+ Visualisation en temps réel avec matplotlib
+ Export Excel
"""
from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings
import simpy
import numpy as np
import pandas as pd
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import ttk, messagebox

# ============================================================
# Modèle radio Beyrouth Rural - Bekaa Valley
# ============================================================
N_EXP    = 3.033
PL0      = 111.75
LH       = -6.65
H_ED     = 1.5
F_MHZ    = 868.0
P_TX_DBM = 14.0

SENSITIVITY = {
    'DR8': -138.0,
    'DR9': -137.0,
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
    if p == 1.0:
        return 'h=2 fixe'
    elif p == 0.0:
        return 'h=3 fixe'
    else:
        return f'p2={p:.2f}/p3={1-p:.2f}'

# ============================================================
# Classes simulation (identiques au script original)
# ============================================================
class FragmentWithRadio(Fragment):
    def __init__(self, type, duration, channel, packet):
        super().__init__(type, duration, channel, packet)
        self.radio_lost = False

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

class NodeWithRadio(Node):
    def __init__(self, obw, headers, payloads, header_duration,
                 payload_duration, transceiver_wait, traffic_generator, per):
        super().__init__(obw, headers, payloads, header_duration,
                         payload_duration, transceiver_wait, traffic_generator)
        self.per = per
        self.packet = PacketWithRadio(self.id, obw, headers, payloads,
                                      header_duration, payload_duration)

    def end_of_transmission(self):
        self.packet = PacketWithRadio(self.id, self.obw, self.headers,
                                      self.payloads, self.header_duration,
                                      self.payload_duration)

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

def run_sim(settings, distance_km=1.0, dr='DR8', h_ed=H_ED, seed=0):
    if not is_covered(distance_km, dr, h_ed):
        return None, None
    per = packet_error_rate(distance_km, dr, h_ed)
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    if settings.base == 'acrda':
        bs = BaseACRDAWithRadio(settings.obw, settings.window_size,
                                settings.window_step, settings.time_on_air,
                                settings.threshold)
        env.process(bs.sic_window(env))
    else:
        bs = Base(settings.obw, settings.threshold)
    nodes = []
    for i in range(settings.number_nodes):
        node = NodeWithRadio(settings.obw, settings.headers, settings.payloads,
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

def run_sim_mixed(number_nodes, distance_km=1.0, dr='DR8', p=0.5,
                  base='acrda', code='1/3', simulation_time=3600,
                  payload_size=10, obw=35, h_ed=H_ED, seed=0):
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
                             s2.transceiver_wait, s2.traffic_generator, per=per)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))
    for _ in range(n3):
        node = NodeWithRadio(s3.obw, s3.headers, s3.payloads,
                             s3.header_duration, s3.payload_duration,
                             s3.transceiver_wait, s3.traffic_generator, per=per)
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
# Interface graphique tkinter
# ============================================================
class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LR-FHSS Sim — Beyrouth Rural (Bekaa Valley)")
        self.root.resizable(False, False)

        # Couleurs
        BG      = "#1e2761"
        FG      = "white"
        ACCENT  = "#029aa0"
        BTN_BG  = "#029aa0"
        ENTRY_BG= "#2c3e6e"

        self.root.configure(bg=BG)

        # ---- Titre ----
        tk.Label(root, text="🛰  LR-FHSS Sim — Beyrouth Rural",
                 font=("Arial", 14, "bold"), bg=BG, fg=ACCENT).grid(
                 row=0, column=0, columnspan=4, pady=(12,4))
        tk.Label(root, text="El Chall, Lahoud, El Helou — IEEE IoT 2019",
                 font=("Arial", 9, "italic"), bg=BG, fg="lightgray").grid(
                 row=1, column=0, columnspan=4, pady=(0,10))

        def label(text, row, col):
            tk.Label(root, text=text, bg=BG, fg=FG,
                     font=("Arial", 10)).grid(row=row, column=col,
                     sticky='e', padx=8, pady=4)

        def entry(default, row, col, width=10):
            e = tk.Entry(root, width=width, bg=ENTRY_BG, fg=FG,
                         insertbackground=FG, font=("Arial", 10),
                         relief='flat', bd=4)
            e.insert(0, str(default))
            e.grid(row=row, column=col, padx=8, pady=4, sticky='w')
            return e

        # ---- Paramètres réseau ----
        tk.Label(root, text="── Paramètres réseau ──",
                 bg=BG, fg=ACCENT, font=("Arial", 10, "bold")).grid(
                 row=2, column=0, columnspan=4, pady=(6,2))

        label("DR :", 3, 0)
        self.dr_var = tk.StringVar(value='DR8')
        ttk.Combobox(root, textvariable=self.dr_var,
                     values=['DR8', 'DR9'], width=8,
                     state='readonly').grid(row=3, column=1, sticky='w', padx=8)

        label("Nœuds réseau :", 3, 2)
        self.nodes_entry = entry(25000, 3, 3)

        label("Base :", 4, 0)
        self.base_var = tk.StringVar(value='acrda')
        ttk.Combobox(root, textvariable=self.base_var,
                     values=['acrda', 'core'], width=8,
                     state='readonly').grid(row=4, column=1, sticky='w', padx=8)

        label("Code rate :", 4, 2)
        self.code_var = tk.StringVar(value='1/3')
        ttk.Combobox(root, textvariable=self.code_var,
                     values=['1/3', '2/3', '1/2', '5/6'], width=8,
                     state='readonly').grid(row=4, column=3, sticky='w', padx=8)

        label("Seed :", 5, 0)
        self.seed_entry = entry(0, 5, 1, width=8)

        label("h_ED (m) :", 5, 2)
        self.hed_entry = entry(1.5, 5, 3, width=8)

        # ---- Distances ----
        tk.Label(root, text="── Distances à tester (km, séparées par virgules) ──",
                 bg=BG, fg=ACCENT, font=("Arial", 10, "bold")).grid(
                 row=6, column=0, columnspan=4, pady=(8,2))

        self.dist_entry = entry("1,10,12,14,16,18,20,22", 7, 0, width=40)
        self.dist_entry.grid(row=7, column=0, columnspan=4, padx=8, pady=4)

        # ---- Distributions ----
        tk.Label(root, text="── Distributions de headers (proportions p2) ──",
                 bg=BG, fg=ACCENT, font=("Arial", 10, "bold")).grid(
                 row=8, column=0, columnspan=4, pady=(8,2))

        self.prop_vars = {}
        props = [
            (1.0,  'h=2 fixe (p2=1.0)'),
            (0.0,  'h=3 fixe (p2=0.0)'),
            (0.5,  '50/50   (p2=0.5)'),
            (0.75, '75/25   (p2=0.75)'),
            (0.25, '25/75   (p2=0.25)'),
        ]
        for i, (p, lbl) in enumerate(props):
            var = tk.BooleanVar(value=True)
            self.prop_vars[p] = var
            tk.Checkbutton(root, text=lbl, variable=var,
                           bg=BG, fg=FG, selectcolor=ENTRY_BG,
                           activebackground=BG, activeforeground=FG,
                           font=("Arial", 10)).grid(
                           row=9 + i//3, column=i%3,
                           sticky='w', padx=12, pady=2)

        # ---- Export Excel ----
        tk.Label(root, text="── Options ──",
                 bg=BG, fg=ACCENT, font=("Arial", 10, "bold")).grid(
                 row=11, column=0, columnspan=4, pady=(8,2))

        self.export_var = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Exporter vers Excel",
                       variable=self.export_var,
                       bg=BG, fg=FG, selectcolor=ENTRY_BG,
                       activebackground=BG, activeforeground=FG,
                       font=("Arial", 10)).grid(
                       row=12, column=0, columnspan=2,
                       sticky='w', padx=12)

        # ---- Barre de progression ----
        self.progress_label = tk.Label(root, text="En attente...",
                                       bg=BG, fg="lightgray",
                                       font=("Arial", 9))
        self.progress_label.grid(row=13, column=0, columnspan=4, pady=(8,2))

        self.progress = ttk.Progressbar(root, length=380, mode='determinate')
        self.progress.grid(row=14, column=0, columnspan=4, padx=12, pady=4)

        # ---- Bouton lancer ----
        self.btn = tk.Button(root, text="▶  Lancer la simulation",
                             command=self.launch,
                             bg=BTN_BG, fg="white",
                             font=("Arial", 11, "bold"),
                             relief='flat', padx=16, pady=6,
                             cursor='hand2')
        self.btn.grid(row=15, column=0, columnspan=4, pady=12)

        # Afficher les infos du modèle
        info = (f"Modèle : Beyrouth Rural  |  n={N_EXP}  "
                f"PL0={PL0} dB  |  f={F_MHZ} MHz  |  P_tx={P_TX_DBM} dBm")
        tk.Label(root, text=info, bg=BG, fg="lightgray",
                 font=("Arial", 8)).grid(row=16, column=0,
                 columnspan=4, pady=(0,8))

    def launch(self):
        # Récupérer les paramètres
        try:
            DR      = self.dr_var.get()
            N_NET   = int(self.nodes_entry.get())
            N_NODES = N_NET // 8
            BASE    = self.base_var.get()
            CODE    = self.code_var.get()
            SEED    = int(self.seed_entry.get())
            H_ED_M  = float(self.hed_entry.get())
            distances = [float(x.strip())
                         for x in self.dist_entry.get().split(',')]
            PROPORTIONS = [p for p, var in self.prop_vars.items() if var.get()]
            if not PROPORTIONS:
                messagebox.showerror("Erreur",
                    "Sélectionnez au moins une distribution !")
                return
            PROPORTIONS.sort(reverse=True)
        except ValueError as e:
            messagebox.showerror("Erreur de saisie", str(e))
            return

        DISTRIBUTIONS = [(p, make_label(p)) for p in PROPORTIONS]
        STYLES = {
            'h=2 fixe'        : ('blue',   'o',  '-'),
            'h=3 fixe'        : ('red',    's',  '-'),
            'p2=0.50/p3=0.50' : ('green',  '^',  '--'),
            'p2=0.75/p3=0.25' : ('orange', 'D',  '--'),
            'p2=0.25/p3=0.75' : ('purple', '*',  '--'),
        }

        total_sims = len(distances) * len(DISTRIBUTIONS)
        self.progress['maximum'] = total_sims
        self.progress['value']   = 0
        self.btn.config(state='disabled', text="⏳ Simulation en cours...")

        # ---- Graphique temps réel ----
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'Beyrouth Rural — {N_NET} nœuds, {DR}, {BASE}, CR={CODE}, h_ED={H_ED_M}m',
            fontsize=12, fontweight='bold')

        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Taux de succès (%)')
        ax1.set_title('Succès vs Distance')
        ax1.set_ylim(0, 105)
        ax1.set_xlim(min(distances)-0.5, max(distances)+0.5)
        ax1.grid(True, alpha=0.4)

        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Goodput (bytes)')
        ax2.set_title('Goodput vs Distance')
        ax2.set_xlim(min(distances)-0.5, max(distances)+0.5)
        ax2.grid(True, alpha=0.4)

        lines1, lines2 = {}, {}
        results = {label: {'x': [], 'succ': [], 'good': []}
                   for _, label in DISTRIBUTIONS}

        for _, label in DISTRIBUTIONS:
            color, marker, ls = STYLES.get(label, ('black', 'o', '-'))
            l1, = ax1.plot([], [], color=color, marker=marker,
                           linestyle=ls, linewidth=2, markersize=6, label=label)
            l2, = ax2.plot([], [], color=color, marker=marker,
                           linestyle=ls, linewidth=2, markersize=6, label=label)
            lines1[label] = l1
            lines2[label] = l2

        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

        # Settings fixes
        s2 = Settings(number_nodes=N_NODES, simulation_time=3600,
                      payload_size=10, headers=2, code=CODE, obw=35, base=BASE)
        s3 = Settings(number_nodes=N_NODES, simulation_time=3600,
                      payload_size=10, headers=3, code=CODE, obw=35, base=BASE)

        start_time = time()
        rows = []
        done = 0

        for d in distances:
            per_val  = packet_error_rate(d, DR, H_ED_M)
            rssi_val = rssi(d, H_ED_M)

            row = {
                'Distance (km)': d,
                'RSSI (dBm)'   : round(rssi_val, 1),
                'PER (%)'      : round(per_val * 100, 1),
            }

            self.progress_label.config(
                text=f"Distance {d} km | RSSI={rssi_val:.1f} dBm | PER={per_val*100:.1f}%")
            self.root.update()

            if not is_covered(d, DR, H_ED_M):
                for p, label in DISTRIBUTIONS:
                    row[f'{label} Succès (%)']  = None
                    row[f'{label} Goodput (B)'] = None
                    done += 1
                    self.progress['value'] = done
                    self.root.update()
                rows.append(row)
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

                done += 1
                self.progress['value'] = done

                if r is not None:
                    results[label]['x'].append(d)
                    results[label]['succ'].append(r * 100)
                    results[label]['good'].append(g)
                    lines1[label].set_data(results[label]['x'],
                                           results[label]['succ'])
                    lines2[label].set_data(results[label]['x'],
                                           results[label]['good'])
                    ax1.relim(); ax1.autoscale_view()
                    ax2.relim(); ax2.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)

                    row[f'{label} Succès (%)']  = round(r * 100, 2)
                    row[f'{label} Goodput (B)'] = g
                else:
                    row[f'{label} Succès (%)']  = None
                    row[f'{label} Goodput (B)'] = None

                self.progress_label.config(
                    text=f"[{done}/{total_sims}] {label} @ {d}km → "
                         f"{r*100:.2f}%" if r else f"[{done}/{total_sims}] hors portée")
                self.root.update()

            rows.append(row)

        # ---- Terminé ----
        elapsed = time() - start_time
        fig.suptitle(
            f'Beyrouth Rural — {N_NET} nœuds, {DR}, {BASE}  ✅ Terminé',
            fontsize=12, fontweight='bold', color='green')
        fig.canvas.draw()
        plt.ioff()

        self.progress_label.config(
            text=f"✅ Terminé en {elapsed:.1f}s", fg='lightgreen')
        self.btn.config(state='normal', text="▶  Lancer la simulation")

        # ---- Export Excel ----
        if self.export_var.get():
            df = pd.DataFrame(rows)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = (f"resultats_bey_rural_{DR}_"
                         f"{N_NET}noeuds_{timestamp}.xlsx")
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Résultats', index=False)
                params_data = {
                    'Paramètre': [
                        'Modèle', 'DR', 'Nœuds réseau', 'Base',
                        'Code rate', 'Seed', 'h_ED (m)',
                        'Portée DR8 (km)', 'Distributions'
                    ],
                    'Valeur': [
                        'Beyrouth Rural (Bekaa)', DR, N_NET, BASE,
                        CODE, SEED, H_ED_M,
                        round(max_distance(DR, H_ED_M), 2),
                        str([label for _, label in DISTRIBUTIONS])
                    ]
                }
                pd.DataFrame(params_data).to_excel(
                    writer, sheet_name='Paramètres', index=False)
            messagebox.showinfo("Export",
                                f"✅ Résultats exportés :\n{filename}")

        plt.show()


# ============================================================
# Lancement
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = SimulationGUI(root)
    root.mainloop()