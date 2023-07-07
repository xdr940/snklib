'''
Author: Wang Xt
Implement association between satellite and ground-station/aerocraft.
The code is a little bit of a shit mountain, but it works fine. Hope improve it later.
'''

import pymap3d as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as interp
from tqdm import tqdm
from datetime import datetime

from ..utils import eci2ecef_np
from ..common import *

class Access:
    def __init__(
            self,
            start_time=None,
            end_time=None,
            time_step=None,
            borderDistance=None,
            dynamic_threshold=False
    ):
        self.start_time = start_time or datetime(2000,1,1)
        self.end_time = end_time or datetime(2000,1,2)
        self.time_step = time_step or 300

        self.duration_sec = int((end_time - start_time).total_seconds())
        self.num_stamps = int((end_time - start_time).total_seconds()/time_step)
        self.time_stamps = np.linspace(0,self.duration_sec,self.num_stamps+1)


        if borderDistance:
            self.borderDistance = borderDistance
        else:
            self.borderDistance = 500000/np.cos(np.deg2rad(50))# 500km high, 40d evelation of sat


        self.df = pd.DataFrame(index=self.time_stamps)
        self.ranges={}
        self.sat_funcs={}
        self.gs_funcs={}
        self.access_stamps={}
        self.sat_positions={}
        self.full_time = np.linspace(start=0, stop=self.duration_sec, num=self.duration_sec+1)


        #post configs (dynamic border)

        self.Re = Re # earth radius
        num_orbits = 20
        h =550000
        # self.max_border = 2 *3.14*(self.Re + h)/num_orbits
        self.max_border = 2000000

        self.dynamic_threshold = dynamic_threshold



    def load_sat(self,sat):
        position = sat['position']['cartesian']
        sat_position = np.array(position).reshape([self.num_stamps+1,4])#292 x 300 = 87600
        # eci 2 ecef build gsls,asls
        sat_position = eci2ecef_np(sat_position)


        self.sat_fx = interp(self.time_stamps, sat_position[:,1], 'cubic')
        self.sat_fy = interp(self.time_stamps, sat_position[:,2], 'cubic')
        self.sat_fz = interp(self.time_stamps, sat_position[:,3], 'cubic')
            # self.sat_funcs[sat['id']] = (fx,fy,fz)


        self.sat_id = sat['id']

        pass
    def load_gs(self,gs):
        self.gs_position = np.array(gs['position']['cartesian'])

        # self.gs_position_homo = [self.gs_position] * 293
        # self.gs_position_homo = np.array(self.gs_position_homo)
        self.gs_id = gs['id']
    def load_ms(self,ms):
        start_time , end_time= ms['availability'].split('/')

        start_time =  datetime.fromisoformat(start_time)
        end_time =  datetime.fromisoformat(end_time)


        duration_sec = (end_time-start_time).seconds
        self.duration_sec = duration_sec
        length = int(len( np.array(ms['position']['cartesian']))/4)
        ms_positions = np.array(ms['position']['cartesian']).reshape([length,4])

        time_stamp = ms_positions[:,0]
        pass
        ms_fx = interp(time_stamp, ms_positions[:, 1], 'cubic')
        ms_fy = interp(time_stamp, ms_positions[:, 2], 'cubic')
        ms_fz = interp(time_stamp, ms_positions[:, 3], 'cubic')

        self.full_time = np.linspace(start=0,stop=duration_sec,num=duration_sec+1)

        fullx = ms_fx(self.full_time)
        fully = ms_fy(self.full_time)
        fullz = ms_fz(self.full_time)
        fullx = np.expand_dims(fullx,0)
        fully = np.expand_dims(fully,0)
        fullz = np.expand_dims(fullz,0)
        self.ms_pos = np.concatenate([fullx,fully,fullz],0).T


        self.ms_id = ms['id']

    def range_log_ms(self):

        fullx = self.sat_fx(self.full_time)
        fully = self.sat_fy(self.full_time)
        fullz = self.sat_fz(self.full_time)
        sat_position = np.concatenate([np.expand_dims(fullx,1),np.expand_dims(fully,1),np.expand_dims(fullz,1)],1)


        ref = sat_position - self.ms_pos
        dis = ref ** 2
        dis = np.sum(dis, axis=1)
        dis = dis ** 0.5
        # tmp/=1000
        mask = dis < self.borderDistance
        if np.sum(mask) > 0:  # range in
            range_name = (self.ms_id, self.sat_id)
            # self.ranges[range_name] = tmp

            # caculate in out instant
            mask = np.int8(mask)
            mask = mask[:-1] - mask[1:]
            start_mask = mask == -1
            end_mask = mask == 1
            starts = list(self.full_time[:-1][start_mask])
            ends = list(self.full_time[:-1][end_mask])

            if len(starts) < len(ends):  # 开始即接入
                starts.insert(0, 0)
            elif len(starts) > len(ends):  # 结束还在接入
                ends.append(self.duration_sec)

            starts = np.expand_dims(np.array(starts), 0)
            ends = np.expand_dims(np.array(ends), 0)

            self.access_stamps[range_name] = np.concatenate([starts, ends], 0).T

    def range_log_gs(self):
        '''
        interp, main calculation
        :return:
        '''



        fullx = self.sat_fx(self.full_time)
        fully = self.sat_fy(self.full_time)
        fullz = self.sat_fz(self.full_time)

        sat_position = np.concatenate([np.expand_dims(fullx,1),np.expand_dims(fully,1),np.expand_dims(fullz,1)],1)

        gs_homo = np.ones_like(sat_position) * self.gs_position

        # distance between sat and gs
        ref = sat_position - gs_homo
        dis = ref**2
        dis = np.sum(dis,axis=1)
        dis = dis**0.5
        # tmp/=1000
        mask = dis<self.borderDistance
        if np.sum(mask)>0:# range in
            range_name = (self.gs_id,self.sat_id)
            # self.ranges[range_name] = tmp

            #caculate in out instant
            mask = np.int8(mask)
            mask = mask[:-1] - mask[1:]
            start_mask = mask == -1
            end_mask = mask == 1
            starts = list(self.full_time[:-1][start_mask])
            ends = list(self.full_time[:-1][end_mask])

            if len(starts) < len(ends):  # 开始即接入
                starts.insert(0, 0)
            elif len(starts) > len(ends):  # 结束还在接入
                ends.append(self.duration_sec)

            starts = np.expand_dims(np.array(starts), 0)
            ends = np.expand_dims(np.array(ends), 0)

            self.access_stamps[range_name] = np.concatenate([starts, ends], 0).T

    def load_sat_rem(self, sat):
        '''

        :param sat:
        :return:
        '''
        position = sat['position']['cartesian']
        sat_position = np.array(position).reshape([self.num_stamps+1, 4])  # 292 x 300 = 87600
        # eci 2 ecef build gsls,asls
        sat_position = eci2ecef_np(sat_position)

        sat_fx = interp(self.time_stamps, sat_position[:, 1], 'cubic')
        sat_fy= interp(self.time_stamps, sat_position[:, 2], 'cubic')
        sat_fz = interp(self.time_stamps, sat_position[:, 3], 'cubic')
        fullx = sat_fx(self.full_time)
        fully = sat_fy(self.full_time)
        fullz = sat_fz(self.full_time)
        self.sat_positions[sat['name']] = np.concatenate([np.expand_dims(fullx, 1), np.expand_dims(fully, 1), np.expand_dims(fullz, 1)], 1)


        # self.sat_funcs[sat['id']] = (fx,fy,fz)
        # self.sat_id = sat['id']

    def sat_with_sat_stamp_update(self,sati_name,satj_name):
        '''
        for tISLs
        :return:
        '''

        ref = self.sat_positions[sati_name]- self.sat_positions[satj_name]
        dis = ref ** 2
        dis = np.sum(dis, axis=1)
        dis = dis ** 0.5
        if not self.dynamic_threshold:
            mask = dis <  self.borderDistance
        else:
            borderDistance = DynamicBorder(self.max_border,self.sat_positions[sati_name],self.sat_positions[satj_name])
            mask = dis <  borderDistance

        if np.sum(mask) == 0:  # range in
            return
        else:
            range_name = ("SAT-{}".format(sati_name), "SAT-{}".format(satj_name))
            # self.ranges[range_name] = tmp

            # caculate in out instant
            mask = np.int8(mask)
            mask = mask[:-1] - mask[1:]
            start_mask = mask == -1
            end_mask = mask == 1
            starts = list(self.full_time[:-1][start_mask])
            ends = list(self.full_time[:-1][end_mask])

            if len(starts) < len(ends):  # 开始即接入
                starts.insert(0, 0)
            elif len(starts) > len(ends):  # 结束还在接入
                ends.append(self.duration_sec)

            starts = np.expand_dims(np.array(starts), 0)
            ends = np.expand_dims(np.array(ends), 0)

            self.access_stamps[range_name] = np.concatenate([starts, ends], 0).T

    def sat_with_gses_stamp_update(self,sat,gses):
        '''

        :param sat:
        :param gses:
        :return: self.access_stamps
        '''
        self.load_sat(sat)
        for gs in gses:
            self.load_gs(gs)

            # caculate range between sat,gs
            self.range_log_gs()

    def sat_with_mses_stamp_update(self,sat,mses):
        '''

        :param sat:
        :param mses:
        :return: self.access_stamps
        '''
        self.load_sat(sat)
        for ms in mses:
            self.load_ms(ms)
            self.range_log_ms()# calculate stamp



    def get_access_stamps(self):
        return self.access_stamps


class BeamForming():
    def __init__(self):
        pass


def DynamicBorder(max_thresh,positions1,positions2):
    Re = 6371137
    positions = (positions1+positions2)/2

    deltas = 1- abs(positions[:,-1]/Re)

    deltas= np.sqrt(deltas)
    deltas *=max_thresh

    return deltas