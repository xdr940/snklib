from path import Path

from ..utils import *

base = ""
class Scenario:
    def __init__(self,config):
        self.config = config
        self.scenario_path = Path(self.config['dump_path'])

        # LAYERED DATA
        self.layer_isls=[]
        self.layer_sats=[]
        self.layer_eisls=[]

        start_time = datetime.datetime.strptime(config['start_time'], '%Y-%m-%dT%H:%M:%SZ')
        end_time = datetime.datetime.strptime(config['end_time'], '%Y-%m-%dT%H:%M:%SZ')
        self.duration_sec = int((end_time - start_time).total_seconds())

        for layer_name, layer in config['constellations'].items():

            sats_p = self.scenario_path / "{}_sats.json".format(layer_name)
            isls_p = self.scenario_path/"{}_isls.json".format(layer_name)
            eisls_p = self.scenario_path/"{}_eisls.json".format(layer_name)
            if sats_p.exists():
                self.layer_sats.append( json2dict(sats_p)['sats'])
            else:
                pass#error dir


            if isls_p.exists():
                self.layer_isls.append( json2dict(isls_p)['isls'])

            if eisls_p.exists():
                self.layer_eisls.append( json2dict(eisls_p)['eisls'])
        self.layer_num_eISL=[]
        for layer in self.layer_eisls:
            self.layer_num_eISL.append(len(layer))

        # print('init over')
        # UN-LAYERED DATA
        # mses
        # gses
    def eISL_time_num(self):
        time_count=np.zeros(3600)
        for eisls in self.layer_eisls:
            for access,durations in eisls.items():
                for duration in durations:
                    start,end = duration[0],duration[1]
                    time_count[int(start):int(end)]+=1
        return time_count
    def visible_time_num_persat(self):
        #`TODO
        time_count=np.zeros(3600)
        for eisls in self.layer_eisls:
            for access, durations in eisls.items():
                for duration in durations:
                    start, end = duration[0], duration[1]
                    time_count[int(start):int(end)] += 1
        return time_count
    def eISL_duration(self,LAYER=0):
        '''

        :param LAYER:
        :return:
        durations_total_mat
        duration_avg_mat
        duration_list
        '''
        idxs = list(range(len(self.layer_sats[LAYER])))
        sat2id = dict(zip(self.layer_sats[LAYER], idxs))

        matrix = np.zeros([len(self.layer_sats[LAYER]), len(self.layer_sats[LAYER])])
        duration_total = np.zeros_like(matrix)
        duration_arcs = np.zeros_like(matrix)
        durs_list = []



        for access, durs in self.layer_eisls[0].items():
            src, dst = access.split('-')
            src_id = sat2id[src]
            dst_id = sat2id[dst]


            duration_arcs[src_id, dst_id] += len(durs)
            duration_arcs[dst_id, src_id] += len(durs)
            for start,end in durs:
                if start <=1 or end >= self.duration_sec-1:
                    continue
                duration = end-start
                # if duration ==22:
                    # print('?')
                durs_list.append(duration)




                duration_total[src_id, dst_id] += duration
                duration_total[dst_id, src_id] += duration



        return duration_total,duration_arcs,durs_list

    def eISL_duration3D(self, LAYER=0):
        '''
        :param LAYER:
        :return:
        durations_total_mat
        duration_avg_mat
        duration_list
        '''
        idxs = list(range(len(self.layer_sats[LAYER])))
        sat2id = dict(zip(self.layer_sats[LAYER], idxs))
        time_count=np.zeros(3600)

        mat3d = np.zeros([len(self.layer_sats[LAYER]), len(self.layer_sats[LAYER]),7200],dtype=np.int8)
        Mt = np.zeros_like(mat3d)
        durs_list = []

        for access, durs in self.layer_eisls[0].items():
            src, dst = access.split('-')
            src_id = sat2id[src]
            dst_id = sat2id[dst]
            for dur in durs:
                start,end = int(dur[0]), int(dur[1])
                Mt[src_id, dst_id,start:end]=1
                Mt[dst_id, src_id,start:end]=1
        return Mt
    def time_ISLlength(self):
        '''
            applied in Instance
        :return:
        '''
        pass


class Analysor:
    def __init__(self):
        pass