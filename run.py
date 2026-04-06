from lrfhss.lrfhss_core import *
from lrfhss.acrda import BaseACRDA
from lrfhss.settings import Settings
import simpy

def run_sim(settings: Settings, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    env = simpy.Environment()
    if settings.base=='acrda':
        bs = BaseACRDA(settings.obw, settings.window_size, settings.window_step, settings.time_on_air, settings.threshold)
        env.process(bs.sic_window(env))
    else:
        bs = Base(settings.obw, settings.threshold)
    
    nodes = []
    for i in range(settings.number_nodes):
        node = Node(settings.obw, settings.headers, settings.payloads, settings.header_duration, settings.payload_duration, settings.transceiver_wait, settings.traffic_generator)
        bs.add_node(node.id)
        nodes.append(node)
        env.process(node.transmit(env, bs))
    # start simulation
    env.run(until=settings.simulation_time)

    # after simulation
    success = sum(bs.packets_received.values())
    transmitted = sum(n.transmitted for n in nodes)

    if transmitted == 0: #If no transmissions are made, we consider 100% success as there were no outages
        return 1
    else:
        
        return [[success/transmitted], [success*settings.payload_size], [transmitted]]

    #Get the average success per device, used to plot the CDF 
    #success_per_device = [1 if n.transmitted == 0 else bs.packets_received[n.id]/n.transmitted for n in nodes]
    #return success_per_device
if __name__ == "__main__":
   #s = Settings()
   s = Settings(
        number_nodes    = 125000//8,
        simulation_time = 3600,
        payload_size    = 10,
        headers         = 2,
        code            = '1/3',
        traffic_param   = {'average_interval': 900},
        obw             = 35,
        base            = 'acrda'
        #window_size=4
    )
   #print(number_nodes,code,headers,base)
   print(f"number_nodes    = {s.number_nodes}")
   #print(f"simulation_time = {s.simulation_time} s")
   #print(f"payload_size    = {s.payload_size} bytes")
   print(f"headers         = {s.headers}")
   #print(f"code    = {s.code}")
   #print(f"payloads        = {s.payloads}")
   #print(f"threshold       = {s.threshold}")
   #print(f"obw             = {s.obw}")
   print(f"base            = {s.base}")
   #print(f"time_on_air     = {s.time_on_air:.4f} s")
   print("-" * 40)
   print(run_sim(s))