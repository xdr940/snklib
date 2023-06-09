import numpy as np
from path import Path
import pymap3d.vincenty  as pmv
import pymap3d as pm

from ..utils import *


def get_pkt(pkt_list,pkt_id):
    for pkt in pkt_list:
        if pkt_id == pkt['id']:
            return pkt
    return None

def entity2xISL_id(src_node, src_module ,tgt_node, tgt_module,entity_type=None):
    src_type,src_num = src_node.split('-')
    tgt_type,tgt_num = tgt_node.split('-')

    entity_id = None
    if entity_type == 'FWD':
        entity_id = "FWD-{}-{}".format(src_num, tgt_num)
        return entity_id

    if src_type == 'SAT' and tgt_type =='SAT':
        if tgt_module in ['rx06']: #eISL
            entity_id = "eISL-{}-{}".format(src_num, tgt_num)

        elif src_module in ['tx02','tx03','tx04','tx05']: #sISL
            entity_id = "ISL-{}-{}".format(src_num, tgt_num)

        elif src_module in ['tx00','tx01']: #iISL
            entity_id = "ISL-{}-{}".format(src_num, tgt_num)
    elif src_type == 'GS' and tgt_type == 'SAT' :
        entity_id = "GSL-{}-{}".format(src_num, tgt_num)

    elif src_type == 'SAT' and tgt_type == 'GS':
        entity_id = "GSL-{}-{}".format(tgt_num, src_num)

    elif src_type == 'MS' and tgt_type == 'SAT' :
        entity_id = "GSL-{}-{}".format(src_num, tgt_num)

    elif src_type == 'SAT' and tgt_type == 'MS':
        entity_id = "GSL-{}-{}".format(tgt_num, src_num)
    if entity_id ==None:
        print()
        pass
    return entity_id




def event2animation(instance):
    event_animation_map = instance.comm_config['event_animation_map']
    pkt_type_color_map = instance.comm_config['pkt_type_color_map']
    transiver_adj = instance.comm_config['transiver_adj']


    animation_list = []
    supp_info_dict = {}


    start = datetime.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)

    for event in instance.event_list:
        delta = datetime.timedelta(seconds=float(event['time']))

        pkt_id = event['pkt_id']
        pkt = get_pkt(instance.packet_list, pkt_id)
        pkt_type = pkt['packet fields']['Type']

        # node animation
        if event['type'] in ['start_trans','end_trans','gen_data']:
            animation = {
                "time": "{}Z".format((start + delta).isoformat())
            }

            entity_id, module = event['source module'].split('.')
            animation["entity"] = entity_id
            animation["state"] = event_animation_map[event['type']]
            animation_list.append(animation)




            # eISLs
            if event['type'] == 'start_trans':
                animation = {
                    "time": "{}Z".format((start + delta).isoformat())
                }
                src_node, src_module = event['source module'].split('.')
                tgt_node, tgt_module = transiver_adj[event['target module']].split('.')

                entity_id = entity2xISL_id(src_node, src_module, tgt_node, tgt_module)


                animation["entity"] = entity_id
                animation["state"] = 'ACTIVE'

                animation_list.append(animation)


        elif event['type'] in ['start_rec','end_rec']:

            #  start rec node, end rec node
            animation = {
                "time": "{}Z".format((start + delta).isoformat())
            }

            entity_id, module = event['target module'].split('.')
            animation["entity"] = entity_id
            animation["state"] = event_animation_map[event['type']]
            animation_list.append(animation)



            # fwd , start rec fwd, end rec fwd
            animation = {
                "time": "{}Z".format((start + delta).isoformat())
            }
            src_node, src_module = event['source module'].split('.')
            tgt_node, tgt_module = event['target module'].split('.')

            entity_id = entity2xISL_id(src_node, src_module, tgt_node, tgt_module, entity_type='FWD')
            animation["entity"] = entity_id

            if event['type'] == 'start_rec':
                animation["state"] = pkt_type_color_map[pkt_type]
                animation["pkt_id"] = pkt_id
                supp_info_dict[pkt_id] = "Type:{}".format(pkt_type)

            else:
                animation["pkt_id"] = pkt_id
                animation["state"] = event_animation_map[event['type']]

            animation_list.append(animation)



            # end rec isl
            if event['type'] == 'end_rec':
                animation = {
                    "time": "{}Z".format((start + delta).isoformat())
                }
                src_node, src_module = event['source module'].split('.')
                tgt_node, tgt_module = event['target module'].split('.')

                entity_id = entity2xISL_id(src_node, src_module, tgt_node, tgt_module)

                animation["entity"] = entity_id
                animation["state"] = 'DEACTIVE'
                animation_list.append(animation)



    return animation_list, supp_info_dict

class Instance:
    def __init__(self,dir,delta=10):
        if typeof(dir)=='str':
            dir = Path(dir)
        self.dir=dir

        # for OPNet
        event_list_p= dir/"event_list.json"
        if event_list_p.exists():

            self.event_list = json2dict(event_list_p)

            self.event_list.sort(key=lambda x: x['time'])

        packet_list_p = dir/"packet_list.json"
        if packet_list_p.exists():
            self.packet_list = json2dict(packet_list_p)

            self.packet_list.sort(key=lambda x: x['id'])


        comm_p= dir/"comm.json"
        if comm_p.exists():
            self.comm_config = json2dict(comm_p)

        # header
        header_p = dir / "header.json"
        if header_p.exists():
            self.header = json2dict(header_p)
            self.time_stamps = self.header['time_stamps']


            if "delta" in self.header.keys():
                self.delta = self.header['delta']
            else:
                self.delta = delta


    #LEVEL 0


        # STATIC DATA
        file = dir/'sats.json'
        if file.exists():
            self.SATs = json2dict(file)
            self.matrix = np.zeros([len(self.SATs), len(self.SATs)])
            idxs = list(range(len(self.SATs)))
            self.sat2id = dict(zip(self.SATs, idxs))



        file = dir/'isls.json'
        if file.exists():
            self.ISLs = json2dict(file)




        # TIME VAR DATA
        self.time_position = []
        position_files = dir.files('*_positions.json')
        position_files.sort()
        for file in position_files:
            self.time_position.append(json2dict(file))

        self.time_eISLs = []
        eISLs_files = dir.files("*_eisls.json")+ dir.files("*_eISLs.json")
        eISLs_files.sort()
        for file in eISLs_files:
            self.time_eISLs.append(json2dict(file))


        route_files = dir.files("*_routes.json")
        route_files.sort()
        self.time_routes=[]
        for file in route_files:
            self.time_routes.append(json2dict(file))



    # LEVEL1 for Scenarios
    # instance generated by sta_download procedure, without paths

    def time_eISLpotion(self,fmt='ECI'):
        '''
        input:
            self.time_position,
            self.layer_isls,
            self.time_eISLs,
            self.time_stamps

        complexity:
            T = O(T x num_sats)

        method:
            (sat1_pos + sat2_pos)/2
        :return:

        '''
        time_pos1 = []
        time_pos2 = []
        total_eISL = set()
        for idx, (time_stamp, eISLs, positions) in enumerate(
                zip(self.time_stamps, self.time_eISLs, self.time_position)):
            pos1 = []
            pos2 = []
            if len(eISLs) == 0:
                pos1.append([np.nan, np.nan, np.nan])
                pos2.append([np.nan, np.nan, np.nan])
            else:
                for eISL in eISLs:
                    if eISL not in total_eISL:
                        total_eISL.add(eISL)
                        _, p1, p2 = eISL.split('-')
                        pos1.append(positions[p1])
                        pos2.append(positions[p2])
                    else:
                        pos1.append([np.nan, np.nan, np.nan])
                        pos2.append([np.nan, np.nan, np.nan])
                        continue


            time_pos1.append(np.array(pos1))
            time_pos2.append(np.array(pos2))
        time_pos1 = homogen3D(time_pos1)
        time_pos2 = homogen3D(time_pos2)
        pos = (time_pos1 + time_pos2)/2


        if fmt =='ECI':
            return pos
        elif fmt =='GEO':
            # pos = eci2geo_np(self.time_stamps,pos) # TODO
            return pos



    def time_eISLlength(self):
        '''
        input:
            self.time_position,
            self.layer_isls,
            self.time_eISLs,
            self.time_stamps
        complexity:
            T = O (Tx num_sats)
        calculation:
            (sat1_pos -sat2_pos)**2
        :return:
            TxNx3
        '''
        time_pos1=[]
        time_pos2=[]
        for idx, (time_stamp, eISLs,positions) in enumerate(zip(self.time_stamps, self.time_eISLs,self.time_position)):
            pos1 = []
            pos2 = []
            if len(eISLs) ==0:
                pos1.append([np.nan,np.nan,np.nan])
                pos2.append([np.nan,np.nan,np.nan])
            else:
                for eISL in eISLs:
                    _,p1,p2 = eISL.split('-')
                    pos1.append(positions[p1])
                    pos2.append(positions[p2])

            time_pos1.append(np.array(pos1))
            time_pos2.append(np.array(pos2))
        time_pos1 = homogen3D(time_pos1)
        time_pos2 = homogen3D(time_pos2)

        return np.linalg.norm(time_pos1-time_pos2,axis=2)
    def time_ISLlength(self):
        time_iISL_pos1 = []
        time_iISL_pos2 = []
        time_sISL_pos1 = []
        time_sISL_pos2 = []

        for idx, (time_stamp,  positions) in enumerate(
                zip(self.time_stamps,  self.time_position)):
            intra_pos1 = []
            intra_pos2 = []
            inter_pos1 =[]
            inter_pos2 = []
            for ISL in self.ISLs:
                    _, p1, p2 = ISL.split('-')
                    if p1[1:3] == p2[1:3]:
                        intra_pos1.append(positions[p1])
                        intra_pos2.append(positions[p2])
                    else:
                        inter_pos1.append(positions[p1])
                        inter_pos2.append(positions[p2])


            time_iISL_pos1.append(np.array(intra_pos1))
            time_iISL_pos2.append(np.array(intra_pos2))

            time_sISL_pos1.append(np.array(inter_pos1))
            time_sISL_pos2.append(np.array(inter_pos2))

        time_iISL_pos1 = homogen3D(time_iISL_pos1)
        time_iISL_pos2 = homogen3D(time_iISL_pos2)

        time_sISL_pos1 = homogen3D(time_sISL_pos1)
        time_sISL_pos2 = homogen3D(time_sISL_pos2)

        time_iISL_length = np.linalg.norm(time_iISL_pos1 - time_iISL_pos2, axis=2)
        time_sISL_length = np.linalg.norm(time_sISL_pos1 - time_sISL_pos2, axis=2)

        return time_iISL_length, time_sISL_length

    def time_SATangle(self):
        #TODO
        pass

    def time_ISLlength2D(self):
        '''

               :return: Tx[NxN]
               '''
        time_eDistanceT = []
        for idx, (time_stamp, eISLs,position) in enumerate(zip(self.time_stamps, self.time_eISLs,self.time_position)):
            eDistanceT = np.zeros_like(self.matrix)

            for eISL in eISLs:
                _, p1, p2 = eISL.split('-')

                d = np.linalg.norm(np.array(position[p1])- np.array(position[p2]))
                eDistanceT[self.sat2id[p1], self.sat2id[p2]] = d
                eDistanceT[self.sat2id[p2], self.sat2id[p1]] = d


            time_eDistanceT.append(np.expand_dims(eDistanceT, axis=0))
        time_eDistanceT = np.concatenate(time_eDistanceT, axis=0)

        time_eDistanceT = zero2nan(time_eDistanceT)

        return time_eDistanceT

    def time_eMatch(self):
        '''

        :return: Tx[NxN]
        '''
        time_eMatchT =[]
        for idx, (time_stamp, eISLs) in enumerate(zip(self.time_stamps, self.time_eISLs)):
            eMatchT = np.zeros_like(self.matrix)

            for eISL in eISLs:
                _, p1, p2 = eISL.split('-')
                eMatchT[self.sat2id[p1],self.sat2id[p2]]+=1
                eMatchT[self.sat2id[p2],self.sat2id[p1]]+=1

            time_eMatchT.append(np.expand_dims(eMatchT,axis=0))
        time_MatchT = np.concatenate(time_eMatchT,axis=0)
        return time_MatchT

    # ==================procedure contest============
    # LEVEL 1
    # instance generated by xx procedure, with paths
    def propDis(self):
        '''

        :return: dim3 data,TxCxN
        '''
        time_conn_peer_propDis = []
        for idx, (time_stamp, routes) in enumerate(zip(self.time_stamps, self.time_routes)):
            conn_peer_propDis=[]
            for route in routes :
                i = 0
                peer_propDis = []
                while i + 1 < len(route['path']):
                    p1 = np.array(self.time_position[idx][route['path'][i]])
                    p2 = np.array(self.time_position[idx][route['path'][i + 1]])
                    peer_propDis.append(np.linalg.norm(p1 - p2))
                    i += 1
                conn_peer_propDis.append(peer_propDis)

            time_conn_peer_propDis.append(homogen2D(conn_peer_propDis,pad_value=0))
        time_conn_peer_propDis = homogen3D(time_conn_peer_propDis,pad_value=0)
        return time_conn_peer_propDis


    def geoDis(self):
        '''

        :return:dim2 data, TxC
        '''
        time_conn_geoDis = []
        for idx, (time_stamp, routes) in enumerate(zip(self.time_stamps, self.time_routes)):
            conn_srcs = []
            conn_dsts = []
            conn_time_arr = []
            conn_geoDis = []
            for route in routes:
                src, dst = route['src_sat'], route['dst_sat']
                conn_srcs.append(self.time_position[idx][src])
                conn_dsts.append(self.time_position[idx][dst])
                conn_time_arr.append(time_stamp)
            conn_srcs = np.array(conn_srcs)
            conn_dsts = np.array(conn_dsts)
            conn_time_arr = np.array(conn_time_arr)

            srcs = pm.eci2geodetic(conn_srcs[:, 0], conn_srcs[:, 1], conn_srcs[:, 2], conn_time_arr)
            dsts = pm.eci2geodetic(conn_dsts[:, 0], conn_dsts[:, 1], conn_dsts[:, 2], conn_time_arr)
            srcs = np.array(srcs).T
            dsts = np.array(dsts).T

            for route, src, dst in zip(routes, srcs, dsts):
                Lgeo, azimuth_degs = pmv.vdist(src[0], src[1], dst[0], dst[1])



                conn_geoDis.append(Lgeo)

            time_conn_geoDis.append(conn_geoDis)
        time_conn_geoDis = np.array(homogen2D(time_conn_geoDis))
        return time_conn_geoDis

    def hops(self):
        '''

        :return: dim2 data, TxC
        '''
        time_conn_hop =[]
        for routes in  self.time_routes:
            conn_hop=[]
            for route in routes:
                conn_hop.append(len(route['path'])-1)
            time_conn_hop.append(conn_hop)
        return homogen2D(time_conn_hop)

    def reachableMask(self):
        '''
        :return: dim2 data,TxC
        '''
        time_conn_reach = []
        for routes in  self.time_routes:
            conn_hop = []
            for route in routes:
                if route['path'][-1] == route['dst_sat']:
                    conn_hop.append(1)
                else:
                    conn_hop.append(0)

            time_conn_reach.append(conn_hop)
        time_conn_reach = homogen2D(time_conn_reach)

        return time_conn_reach.astype(bool)



    def vertexes(self):
        '''
        low level data,dim3
        :return:
        '''
        time_conn_vertex = []
        for routes in self.time_routes:
            conn_vertex = []
            for route in routes:
                conn_vertex.append(route['path'])

            time_conn_vertex.append(homogen2D(conn_vertex))
        return homogen3D(time_conn_vertex)
    def aptfwd(self):
        '''

        :return: two dim2 data, TxN
        '''
        time_conn_apfwd = []
        time_conn_atfwd = []

        for idx, (time_stamp, routes, eISLs) in enumerate(zip(self.time_stamps, self.time_routes, self.time_eISLs)):
            conn_apfwd = []
            conn_atfwd = []
            for route in routes:
                i = 0
                pfwd_cnt = 0
                tfwd_cnt = 0

                while i + 1 < len(route['path']):
                    p1 = route['path'][i]
                    p2 = route['path'][i + 1]
                    # for layer in self.ISLs:

                    if "tISL-{}-{}".format(p1, p2) in eISLs or "tISL-{}-{}".format(p2, p1) in eISLs:
                        tfwd_cnt += 1
                    elif "ISL-{}-{}".format(p1, p2) in self.ISLs or "ISL-{}-{}".format(p2, p1) in self.ISLs:
                        pfwd_cnt += 1
                    i += 1
                conn_apfwd.append(pfwd_cnt)
                conn_atfwd.append(tfwd_cnt)

            time_conn_apfwd.append(conn_apfwd)
            time_conn_atfwd.append(conn_atfwd)

        return homogen2D(time_conn_apfwd), homogen2D(time_conn_atfwd)



    # Level 2
    def stretch(self):
        '''
        dim 3 data, T C N
        :return:
        '''
        return  self.propDis().sum(axis=-1)/self.geoDis()

    def propLatency(self):
        #mask?
        return self.propDis().sum(-1)/300000


    def pathDiversity(self):
        vs = self.vertexes()
        hops_o_paths = self.hops().mean(-1)
        paths = self.hops()
        pass
    # Level 3
    def throughput(self):
        propDis = self.propDis()
        # mask = self.reachableMask()
        throughputs = []
        for t in range(len(propDis)):
            propDis_t = propDis[t]
            propDis_t = zero2Inf(propDis_t)
            min_dis = propDis_t.min(axis=-1)
            bottleneck = Comm(min_dis)
            bottleneck = np.expand_dims(bottleneck,axis=0)
            throughputs.append(bottleneck)
        throughputs = np.concatenate(throughputs)
        return throughputs
