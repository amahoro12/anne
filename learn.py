import numpy as np
import time
import sys
import tkinter as tk
import numpy as np
import tensorflow as tf

np.random.seed(1)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import random

'''HYPERPARAMETERS'''

# steps = 30000                                      # number of learning steps
start_increase_e_greedy = 10000  # start increse greedy action
k_reward = 10  # reward multiplication
'''net'''
bias_on = True  # on\off  bias (True\False)
n_n = 7  # n_neurons_hidden_layer
l_r = 0.001  # learning_rate
r_w = 0.9  # reward_decay
e_g_enc = 0.001  # e_greedy_ecrement
r_t_i = 1  # replace_target_iter
m_s = 5000  # memory_size
b_s = 32  # mini_batch_size
epsil = 0.9  # start greedy-coefficient

''' enviroment'''
n_f = 1  # count of features: power1, power2, "frequency"........
random_consumption_on = True  # random consumptiot in timeline  no\off  (True\False)

# start parameters:
p_1 = 200
p_2 = 100
c = 300
k_power = 10  # normalisation input power
k_freq = 1  # normalisation input "freq"

max_p_1 = 220  # max power 1 station
min_p_1 = 180  # min power 1 station

max_p_2 = 110  # max power 1 station
min_p_2 = 90  # min power 1 station

max_c = 310  # max power consumption
min_c = 290  # min power 1 station

# act = ['up_power1', 'down_power1', 'up_power2', 'down_power2', 'no actions']         # actions
# act = ['up_power1', 'down_power1']                                                                       # actions
act = ['up_power1', 'down_power1', 'up_power2', 'down_power2']

output_graph = True  # make graph



## This script set up classes for 4 bus and 2 bus environment
import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as plot
import enlopy as el
import numpy as np
import pandas as pd
import pickle
import copy
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandapower.control as ct
import statistics as stat

pd.options.display.float_format = '{:.4g}'.format

### This 4-bus class is not complete as of handover to ABB PG and Magnus Tarle.
# The 2-bus class further below is however complete.
class powerGrid_ieee4:
    def __init__(self, numberOfTimeStepsPerState=4):
        print('in init. Here we lay down the grid structure and load some random state values based on IEEE dataset');
        with open('Data/JanLoadEvery5mins.pkl', 'rb') as pickle_file:
            self.loadProfile = pickle.load(pickle_file)
        with open('Data/generatorValuesEvery5mins.pkl', 'rb') as pickle_file:
            self.powerProfile = pickle.load(pickle_file)

        with open('Data/trainIndices.pkl', 'rb') as pickle_file:
            self.trainIndices = pickle.load(pickle_file)
        with open('Data/testIndices.pkl', 'rb') as pickle_file:
            self.testIndices = pickle.load(pickle_file)

        self.k_old=0;
        self.q_old=0;
        self.actionSpace = {'v_ref_pu': [i*5 / 100 for i in range(16, 25)], 'lp_ref': [i * 5 for i in range(0, 31)]}
        ## Basic ieee 4bus system
        self.net = pp.networks.case4gs();
        ####Shunt FACTS device (bus 1)
        # MV bus
        bus_SVC = pp.create_bus(self.net, name='MV SVCtrafo bus', vn_kv=69, type='n', geodata=(-2, 2.5), zone=2,
                                max_vm_pu=1.1,
                                min_vm_pu=0.9)
        # Trafo
        trafoSVC = pp.create_transformer_from_parameters(self.net, hv_bus=1, lv_bus=4, in_service=True,
                                                         name='trafoSVC', sn_mva=110, vn_hv_kv=230, vn_lv_kv=69,
                                                         vk_percent=12, vkr_percent=0.26, pfe_kw=55, i0_percent=0.06,
                                                         shift_degree=0, tap_side='hv', tap_neutral=0, tap_min=-9,
                                                         tap_max=9,
                                                         tap_step_percent=1.5, tap_step_degree=0,
                                                         tap_phase_shifter=False)
        # Tap changer usually not used on this trafo in real life implementation
        #trafo_control = ct.DiscreteTapControl(net=self.net, tid=0, vm_lower_pu=0.95, vm_upper_pu=1.05)

        # Breaker between grid HV bus and trafo HV bus to connect buses
        sw_SVC = pp.create_switch(self.net, bus=1, element=0, et='t', type='CB', closed=False)
        # Shunt device connected with MV bus
        shuntDev = pp.create_shunt(self.net, bus_SVC, 0, in_service=True, name='Shunt Device', step=1)

        ##Series device (at line 3, in middle between bus 2 and 3)
        # Add intermediate buses for bypass and series compensation impedance
        bus_SC1 = pp.create_bus(self.net, name='SC bus 1', vn_kv=230, type='n', geodata=(-1, 3.1), zone=2,
                                max_vm_pu=1.1, min_vm_pu=0.9)
        bus_SC2 = pp.create_bus(self.net, name='SC bus 2', vn_kv=230, type='n', geodata=(-1, 3.0), zone=2,
                                max_vm_pu=1.1, min_vm_pu=0.9)
        sw_SC_bypass = pp.create_switch(self.net, bus=5, element=6, et='b', type='CB', closed=True)
        imp_SC = pp.create_impedance(self.net, from_bus=5, to_bus=6, rft_pu=0.01272, xft_pu=-0.0636,
                                     rtf_pu=0.01272, xtf_pu=-0.0636, sn_mva=250, in_service=True)
        # Adjust orginal Line 3 to connect to new buses instead.
        self.net.line.at[3, ['length_km', 'to_bus', 'name']] = [0.5, 5, 'line1_SC']

        # Change PV generator to static generator
        self.net.gen.drop(index=[0], inplace=True)  # Drop PV generator
        pp.create_sgen(self.net, 3, p_mw=318, q_mvar=181.4, name='static generator', scaling=1)

        # Randomize starting index in load/gen profiles
        self.numberOfTimeStepsPerState=numberOfTimeStepsPerState;
        self.stateIndex = np.random.randint(len(self.loadProfile)-self.numberOfTimeStepsPerState, size=1)[0];
        #self.stateIndex=0
        self.scaleLoadAndPowerValue(self.stateIndex);
        try:
            pp.runpp(self.net, run_control=False)
            print('Environment has been successfully initialized');
        except:
            print('Some error occured while creating environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    # Power flow calculation, runControl = True gives shunt device trafo tap changer iterative control activated
    def runEnv(self, runControl):
        try:
            pp.runpp(self.net, run_control=runControl);
            #print('Environment has been successfully initialized');
        except:
            print('Some error occurred while creating environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    ## Retreieve voltage and line loading percent as measurements of current state
    def getCurrentState(self):
        bus_index_shunt = 1
        line_index = 1;
        return (self.net.res_bus.vm_pu[bus_index_shunt], self.net.res_line.loading_percent[line_index]);

    ## Retrieve measurements for multiple buses, including load angle for DQN as well
    def getCurrentStateForDQN(self):
        return [self.net.res_bus.vm_pu[1:-3], self.net.res_line.loading_percent[0:], self.net.res_bus.va_degree[1:-3]];

    ## UPDATE NEEED:
    def takeAction(self, lp_ref, v_ref_pu):
        #q_old = 0
        bus_index_shunt = 1
        line_index=3;
        impedenceBackup = self.net.impedance.loc[0, 'xtf_pu'];
        shuntBackup = self.net.shunt.q_mvar
        self.net.switch.at[1, 'closed'] = False
        self.net.switch.at[0, 'closed'] = True

        ##shunt compenstation
        q_comp = self.Shunt_q_comp(v_ref_pu, bus_index_shunt, self.q_old);
        self.q_old = q_comp;
        self.net.shunt.q_mvar =  q_comp;

        ##series compensation
        k_x_comp_pu = self.K_x_comp_pu(lp_ref, 1, self.k_old);
        self.k_old = k_x_comp_pu;
        x_line_pu = self.X_pu(line_index)
        self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']] = x_line_pu * k_x_comp_pu
        networkFailure = False

        self.stateIndex += 1;
        if self.stateIndex < len(self.powerProfile):
            self.scaleLoadAndPowerValue(self.stateIndex);
            try:
                pp.runpp(self.net, run_control=True);
                reward = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
            except:
                print('Unstable environment settings');
                networkFailure = True;
                reward = -1000;

        return (self.net.res_bus.vm_pu[bus_index_shunt], self.net.res_line.loading_percent[line_index]), reward, self.stateIndex == len(self.powerProfile) or networkFailure;
        """
        try:
            pp.runpp(self.net);
            reward = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
        except:
            networkFailure=True;
            self.net.shunt.q_mvar=shuntBackup;
            self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']]=impedenceBackup;
            pp.runpp(self.net);
            reward=1000;
            return self.net.res_bus,reward,True;
        self.stateIndex += 1;
        if self.stateIndex < len(self.powerProfile):
            if (self.scaleLoadAndPowerValue(self.stateIndex, self.stateIndex - 1) == False):
                networkFailure = True;
                reward = 1000;
                # self.stateIndex -= 1;
        return self.net.res_bus, reward, self.stateIndex == len(self.powerProfile) or networkFailure;
        """
    ##Function to calculate line reactance in pu
    def X_pu(self, line_index):
        s_base = 100e6
        v_base = 230e3
        x_base = pow(v_base, 2) / s_base
        x_line_ohm = self.net.line.x_ohm_per_km[line_index]
        x_line_pu = x_line_ohm / x_base  # Can take one since this line is divivded into
        # 2 identical lines with length 0.5 km
        return x_line_pu

    def reset(self):
        print('reset the current environment for next episode');
        oldIndex = self.stateIndex;
        self.stateIndex = np.random.randint(len(self.loadProfile)-1, size=1)[0];
        self.net.switch.at[0, 'closed'] = False
        self.net.switch.at[1, 'closed'] = True
        self.k_old = 0;
        self.q_old = 0;
        self.scaleLoadAndPowerValue(self.stateIndex);
        try:
            pp.runpp(self.net, run_control=False);
            print('Environment has been successfully initialized');
        except:
            print('Some error occurred while resetting the environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    # Calculate immediate reward with loadangle as optional
    def calculateReward(self, voltages, loadingPercent, loadAngles=10):
        try:
            rew = 0;
            for i in range(1, len(voltages)-2): # Dont need to include bus 0 as it is the slack with constant voltage and angle
                                                # -2 because dont want to inclue buses created for FACTS device implementation (3 of them)
                if voltages[i] > 1.25 or voltages[i] < 0.8:
                    rew -= 50;
                elif voltages[i] > 1.1 or voltages[i] < 0.9:
                    rew -= 25;
                elif voltages[i] > 1.05 or voltages[i] < 0.95:
                    rew -= 10;
                elif voltages[i] > 1.025 or voltages[i] < 0.975:
                    rew += 10;
                else:
                    rew += 20;
            rew = rew
            loadingPercentInstability = np.std(loadingPercent) * len(loadingPercent);
            rew -= loadingPercentInstability
            # Check load angle
            for i in range(1, len(loadAngles)-2):
                if abs(loadAngles[i]) >= 30:
                    rew -= 200
        except:
            print('exception in calculate reward')
            print(voltages);
            print(loadingPercent)
            return 0;
        return rew

    ## Simple plot of one-line diagram
    def plotGridFlow(self):
        print('plotting powerflow for the current state')
        plot.simple_plot(self.net)

    ## Scale load and generation from load and generation profiles
    ## Update Needed (Nominal Values)
    def scaleLoadAndPowerValue(self,index):
        scalingFactorLoad = self.loadProfile[index] / (sum(self.loadProfile)/len(self.loadProfile));
        scalingFactorPower = self.powerProfile[index] / max(self.powerProfile);

        # Scaling all loads and the static generator
        self.net.load.p_mw = self.net.load.p_mw * scalingFactorLoad;
        self.net.load.q_mvar = self.net.load.q_mvar * scalingFactorLoad;
        self.net.sgen.p_mw = self.net.sgen.p_mw * scalingFactorPower;
        self.net.sgen.q_mvar = self.net.sgen.q_mvar * scalingFactorPower;

    ## UPDATE NEEDED:
    ##Function for transition from reference power to reactance of series device
    def K_x_comp_pu(self, loading_perc_ref, line_index, k_old):
        ##NEW VERSION TEST:
        c = 15  # Coefficient for transition
        k_x_comp_max_ind = 0.4
        k_x_comp_max_cap = -k_x_comp_max_ind
        loading_perc_meas = self.net.res_line.loading_percent[line_index]
        k_delta = (c * k_x_comp_max_ind * (
                    loading_perc_meas - loading_perc_ref) / 100) - k_old  # 100 To get percentage in pu
        k_x_comp = k_delta + k_old

        # Bypassing series device if impedance close to 0
        if abs(k_x_comp) < 0.0001:  # Helping with convergence
            self.net.switch.closed[1] = True  # ACTUAL network, not a copy

        # Make sure output within rating of device
        if k_x_comp > k_x_comp_max_ind:
            k_x_comp = k_x_comp_max_ind
        if k_x_comp < k_x_comp_max_cap:
            k_x_comp = k_x_comp_max_cap
        return k_x_comp

    ## UPDATE NEEDED:
    ## Function for transition from reference parameter to reactive power output of shunt device
    def Shunt_q_comp(self, v_ref_pu, bus_index, q_old):
        v_bus_pu = self.net.res_bus.vm_pu[bus_index]
        k = 25  # Coefficient for transition, tuned to hit 1 pu with nominal IEEE
        q_rated = 100  # Mvar
        q_min = -q_rated
        q_max = q_rated
        q_delta = k * q_rated * (
                    v_bus_pu - v_ref_pu) - q_old  # q_old might come in handy later with RL if able to take actions without
        # independent change in environment
        q_comp = q_delta + q_old

        if q_comp > q_max:
            q_comp = q_max
        if q_comp < q_min:
            q_comp = q_min

        # print(q_comp)
        return q_comp


#The class for the 2-bus test network used in the Master Thesis by Joakim Oldeen & Vishnu Sharma.
#The class also include several methods used by different RL algorithms such as taking action, calculating reward, recieving states and more
class powerGrid_ieee2:
    def __init__(self,method):
        #print('in init. Here we lay down the grid structure and load some random state values based on IEEE dataset');
        self.method=method;
        if self.method in ('dqn','ddqn','td3'):
            self.errorState=[-2, -1000, -90];
            self.numberOfTimeStepsPerState=3
        else:
            self.errorState=[-2,-1000];
            self.numberOfTimeStepsPerState=1
        with open('Data/JanLoadEvery5mins.pkl', 'rb') as pickle_file:
            self.loadProfile = pickle.load(pickle_file)
        with open('Data/generatorValuesEvery5mins.pkl', 'rb') as pickle_file:
            self.powerProfile = pickle.load(pickle_file)

        with open('Data/trainIndices.pkl', 'rb') as pickle_file:
            self.trainIndices = pickle.load(pickle_file)
        with open('Data/testIndices.pkl', 'rb') as pickle_file:
            self.testIndices = pickle.load(pickle_file)

        self.testIndices = [860,860,860]
        self.actionSpace = {'v_ref_pu': [i*5 / 100 for i in range(18, 23)], 'lp_ref': [i * 15 for i in range(0, 11)]}
        #self.deepActionSpace = {'v_ref_pu': [i/ 100 for i in range(90, 111)], 'lp_ref': [i * 5 for i in range(0, 31)]}
        self.deepActionSpace = {'v_ref_pu': [i*2/100 for i in range(45, 56)], 'lp_ref': [i * 10 for i in range(0, 16)]}
        self.k_old = 0;
        self.q_old = 0;


        ## Basic ieee 4bus system to copy parts from
        net_temp = pp.networks.case4gs();
        # COPY PARAMETERS FROM TEMP NETWORK TO USE IN 2 BUS RADIAL SYSTEM.
        # BUSES
        b0_in_service = net_temp.bus.in_service[0]
        b0_max_vm_pu = net_temp.bus.max_vm_pu[0]
        b0_min_vm_pu = net_temp.bus.min_vm_pu[0]
        b0_name = net_temp.bus.name[0]
        b0_type = net_temp.bus.type[0]
        b0_vn_kv = net_temp.bus.vn_kv[0]
        b0_zone = net_temp.bus.zone[0]
        b0_geodata = (3, 2)

        b1_in_service = net_temp.bus.in_service[1]
        b1_max_vm_pu = net_temp.bus.max_vm_pu[1]
        b1_min_vm_pu = net_temp.bus.min_vm_pu[1]
        b1_name = net_temp.bus.name[1]
        b1_type = net_temp.bus.type[1]
        b1_vn_kv = net_temp.bus.vn_kv[1]
        b1_zone = net_temp.bus.zone[1]
        b1_geodata = (4, 2)

        # BUS ELEMENTS
        load_bus = net_temp.load.bus[1]
        load_in_service = net_temp.load.in_service[1]
        load_p_mw = net_temp.load.p_mw[1]
        load_q_mvar = net_temp.load.q_mvar[1]
        load_scaling = net_temp.load.scaling[1]

        extGrid_bus = net_temp.ext_grid.bus[0]
        extGrid_in_service = net_temp.ext_grid.in_service[0]
        extGrid_va_degree = net_temp.ext_grid.va_degree[0]
        extGrid_vm_pu = net_temp.ext_grid.vm_pu[0]
        extGrid_max_p_mw = net_temp.ext_grid.max_p_mw[0]
        extGrid_min_p_mw = net_temp.ext_grid.min_p_mw[0]
        extGrid_max_q_mvar = net_temp.ext_grid.max_q_mvar[0]
        extGrid_min_q_mvar = net_temp.ext_grid.min_q_mvar[0]

        # LINES
        line0_scaling = 1
        line0_c_nf_per_km = net_temp.line.c_nf_per_km[0]
        line0_df = net_temp.line.df[0]
        line0_from_bus = net_temp.line.from_bus[0]
        line0_g_us_per_km = net_temp.line.g_us_per_km[0]
        line0_in_service = net_temp.line.in_service[0]
        line0_length_km = net_temp.line.length_km[0]
        line0_max_i_ka = net_temp.line.max_i_ka[0]
        line0_max_loading_percent = net_temp.line.max_loading_percent[0]
        line0_parallel = net_temp.line.parallel[0]
        line0_r_ohm_per_km = net_temp.line.r_ohm_per_km[0] * line0_scaling
        line0_to_bus = net_temp.line.to_bus[0]
        line0_type = net_temp.line.type[0]
        line0_x_ohm_per_km = net_temp.line.x_ohm_per_km[0] * line0_scaling

        line1_scaling = 1.2
        line1_c_nf_per_km = line0_c_nf_per_km
        line1_df = line0_df
        line1_from_bus = line0_from_bus
        line1_g_us_per_km = line0_g_us_per_km
        line1_in_service = line0_in_service
        line1_length_km = line0_length_km
        line1_max_i_ka = line0_max_i_ka
        line1_max_loading_percent = line0_max_loading_percent
        line1_parallel = line0_parallel
        line1_r_ohm_per_km = line0_r_ohm_per_km
        line1_to_bus = line0_to_bus
        line1_type = line0_type
        line1_x_ohm_per_km = line0_x_ohm_per_km * line1_scaling # Assume that the lines are identical except for line reactance

        ## creating 2 bus system using nominal values from 4 bus system
        self.net = pp.create_empty_network()
        # Create buses
        b0 = pp.create_bus(self.net, in_service=b0_in_service, max_vm_pu=b0_max_vm_pu, min_vm_pu=b0_min_vm_pu,
                           name=b0_name, type=b0_type, vn_kv=b0_vn_kv, zone=b0_zone, geodata=b0_geodata)

        b1 = pp.create_bus(self.net, in_service=b1_in_service, max_vm_pu=b1_max_vm_pu, min_vm_pu=b1_min_vm_pu,
                           name=b1_name, type=b1_type, vn_kv=b1_vn_kv, zone=b1_zone, geodata=b1_geodata)

        # Create bus elements
        load = pp.create_load(self.net, bus=load_bus, in_service=load_in_service,
                              p_mw=load_p_mw, q_mvar=load_q_mvar, scaling=load_scaling)

        extGrid = pp.create_ext_grid(self.net, bus=extGrid_bus, in_service=extGrid_in_service,
                                     va_degree=extGrid_va_degree,
                                     vm_pu=extGrid_vm_pu, max_p_mw=extGrid_max_p_mw, min_p_mw=extGrid_min_p_mw,
                                     max_q_mvar=extGrid_max_q_mvar, min_q_mvar=extGrid_min_q_mvar)

        # Create lines
        l0 = pp.create_line_from_parameters(self.net, c_nf_per_km=line0_c_nf_per_km, df=line0_df, from_bus=line0_from_bus,
                                            g_us_per_km=line0_g_us_per_km, in_service=line0_in_service,
                                            length_km=line0_length_km,
                                            max_i_ka=line0_max_i_ka, max_loading_percent=line0_max_loading_percent,
                                            parallel=line0_parallel, r_ohm_per_km=line0_r_ohm_per_km,
                                            to_bus=line0_to_bus,
                                            type=line0_type, x_ohm_per_km=line0_x_ohm_per_km)

        l1 = pp.create_line_from_parameters(self.net, c_nf_per_km=line1_c_nf_per_km, df=line1_df, from_bus=line1_from_bus,
                                            g_us_per_km=line1_g_us_per_km, in_service=line1_in_service,
                                            length_km=line1_length_km,
                                            max_i_ka=line1_max_i_ka, max_loading_percent=line1_max_loading_percent,
                                            parallel=line1_parallel, r_ohm_per_km=line1_r_ohm_per_km,
                                            to_bus=line1_to_bus,
                                            type=line1_type, x_ohm_per_km=line1_x_ohm_per_km)

        ####Shunt FACTS device (bus 1)
        # MV bus
        bus_SVC = pp.create_bus(self.net, name='MV SVCtrafo bus', vn_kv=69, type='n', geodata=(4.04, 1.98), zone=2,
                                max_vm_pu=1.1,
                                min_vm_pu=0.9)
        # Trafo
        trafoSVC = pp.create_transformer_from_parameters(self.net, hv_bus=1, lv_bus=2, in_service=True,
                                                         name='trafoSVC', sn_mva=110, vn_hv_kv=230, vn_lv_kv=69,
                                                         vk_percent=12, vkr_percent=0.26, pfe_kw=55, i0_percent=0.06,
                                                         shift_degree=0, tap_side='hv', tap_neutral=0, tap_min=-9,
                                                         tap_max=9,
                                                         tap_step_percent=1.5, tap_step_degree=0,
                                                         tap_phase_shifter=False)
        # TAP Changer on shunt device usually not used in Real life implementation.
        #trafo_control = ct.DiscreteTapControl(net=self.net, tid=0, vm_lower_pu=0.95, vm_upper_pu=1.05)

        # Breaker between grid HV bus and trafo HV bus to connect buses
        sw_SVC = pp.create_switch(self.net, bus=1, element=0, et='t', type='CB', closed=False)
        # Shunt devices connected with MV bus
        shuntDev = pp.create_shunt(self.net, bus_SVC, 2, in_service=True, name='Shunt Device', step=1)

        ####Series device (at line 1, in middle between bus 0 and 1)
        # Add intermediate buses for bypass and series compensation impedance
        bus_SC1 = pp.create_bus(self.net, name='SC bus 1', vn_kv=230, type='n', geodata=(3.48, 2.05),
                                zone=2, max_vm_pu=1.1, min_vm_pu=0.9)
        bus_SC2 = pp.create_bus(self.net, name='SC bus 2', vn_kv=230, type='n', geodata=(3.52, 2.05),
                                zone=2, max_vm_pu=1.1, min_vm_pu=0.9)
        sw_SC_bypass = pp.create_switch(self.net, bus=3, element=4, et='b', type='CB', closed=True)
        imp_SC = pp.create_impedance(self.net, from_bus=3, to_bus=4, rft_pu=0.0000001272, xft_pu=-0.0636*0.4,
                                     rtf_pu=0.0000001272, xtf_pu=-0.0636*0.4, sn_mva=250,
                                     in_service=True)  # Just some default values
        # Adjust orginal Line 3 to connect to new buses instead.
        self.net.line.at[1, ['length_km', 'to_bus', 'name']] = [0.5, 3, 'line1_SC']


        self.nominalP=self.net.load.p_mw[0]
        self.nominalQ=self.net.load.q_mvar[0]



        ## select a random state for the episode
        #self.stateIndex = np.random.randint(len(self.loadProfile)-1-self.numberOfTimeStepsPerState, size=1)[0];

    def setMode(self,mode):
        if mode=='train':
            self.source=self.trainIndices;
        else:
            self.source=self.testIndices;
        self.stateIndex = self.getstartingIndex()
        self.scaleLoadAndPowerValue(self.stateIndex);
        try:
            pp.runpp(self.net, run_control=False);
            print('Environment has been successfully initialized');
            # Create SHUNT controllers
            self.shuntControl = ShuntFACTS(net=self.net, busVoltageInd=1, convLim=0.0005)
            self.seriesControl = SeriesFACTS(net=self.net, lineLPInd=1, convLim=0.0005, x_line_pu=self.X_pu(1))

        except:
            print('Some error occurred while creating environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    def getstartingIndex(self):
        index = np.random.randint(len(self.source), size=1)[0];
        if self.source[index] + self.numberOfTimeStepsPerState < len(self.loadProfile):
            return self.source[index];
        else:
            return self.getstartingIndex()

    # Power flow calculation, runControl = True gives shunt device trafo tap changer iterative control activated
    def runEnv(self, runControl):
        try:
            pp.runpp(self.net, run_control=runControl);
            #print('Environment has been successfully initialized');
        except:
            #print(self.net.load.p_mw[0],self.net.load.q_mvar[0]);
            #print(self.stateIndex)
            #print(len(self.powerProfile))
            if runControl:
                print('Some error occurred while running environment after load increment in runEnv Function in DQN');
            else:
                print('Some error occurred while running environment after reset in runEnv Function in DQN');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    ## Retreieve voltage and line loading percent as measurements of current state
    def getCurrentState(self):
        bus_index_shunt = 1
        line_index = 1;
        return [self.net.res_bus.vm_pu[bus_index_shunt], self.net.res_line.loading_percent[line_index]];

    def getCurrentStateForDQN(self):
        bus_index_shunt = 1
        line_index = 1;
        return [self.net.res_bus.vm_pu[bus_index_shunt], self.net.res_line.loading_percent[line_index]/150, self.net.res_bus.va_degree[bus_index_shunt]/30];

    # Return mean line loading in system. Emulation of what system operator would have set loading reference to.
    def lp_ref_operator(self):
        return stat.mean(self.net.res_line.loading_percent)

    ## Take epsilon-greedy action
    ## Return next state measurements, reward, done (boolean)
    def takeAction(self, lp_ref, v_ref_pu):
        # print('taking action')
        stateAfterAction = copy.deepcopy(self.errorState);
        stateAfterEnvChange = copy.deepcopy(self.errorState);
        measAfterAction = [-2, -1000, -1000]
        self.net.switch.at[0, 'closed'] = True
        self.net.switch.at[1, 'closed'] = False
        if lp_ref != 'na' and v_ref_pu != 'na':
            self.shuntControl.ref = v_ref_pu;
            self.seriesControl.ref = lp_ref;
        networkFailure = False
        done = False;
        bus_index_shunt = 1;
        line_index = 1;
        if self.stateIndex < min(len(self.powerProfile), len(self.loadProfile)):
            try:
                dummyRes = (self.net.res_bus.vm_pu, self.net.res_line.loading_percent)
                ## state = (voltage,ll,angle,p,q)
                pp.runpp(self.net, run_control=True);
                if self.method in ('dqn', 'ddqn','td3'):
                    reward1 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent,
                                                   self.net.res_bus.va_degree[bus_index_shunt]);
                    stateAfterAction = self.getCurrentStateForDQN()
                else:
                    reward1 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
                    stateAfterAction = self.getCurrentState()
                #print('rew1: ', reward1)
                measAfterAction = [self.net.res_bus.vm_pu[1], max(self.net.res_line.loading_percent), np.std(self.net.res_line.loading_percent)]
                done = self.stateIndex == (len(self.powerProfile) - 1)
                if done == False:
                    self.incrementLoadProfile()
                    if self.method in ('dqn', 'ddqn','td3'):
                        reward2 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent,
                                                       self.net.res_bus.va_degree[bus_index_shunt]);
                        stateAfterEnvChange = self.getCurrentStateForDQN()
                    else:
                        reward2 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
                        stateAfterEnvChange = self.getCurrentState()
                #print('rew2: ',reward2)
                reward = 0.7 * reward1 + 0.3 * reward2;
            except:
                print('Unstable environment settings in takeAction(). Action: ', (lp_ref, v_ref_pu), 'p_mw: ', self.net.load.p_mw[0]);
                print('shunt, series, series switch: ', self.net.shunt.q_mvar[0], self.net.impedance.loc[0, ['xft_pu']], self.net.switch.closed[1])
                #print(stateAfterEnvChange)
                #print(stateAfterAction)
                #print(lp_ref,v_ref_pu)
                # print(dummyRes)
                #print(self.net.load.p_mw[0],self.net.load.q_mvar[0]);
                networkFailure = True;
                reward = 0;
                # return stateAfterAction, reward, networkFailure,stateAfterEnvChange ;
        else:
            print('wrong block!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        stateAfterEnvChange.extend(stateAfterAction)
        # print(self.errorState)

        # print(reward2)
        #print('totrew: ', reward)
        return stateAfterEnvChange, reward, done or networkFailure, measAfterAction;

    ## Same as Take Action but without Try for debugging
    def takeAction_noTry(self, lp_ref, v_ref_pu):
        # print('taking action')
        stateAfterAction = copy.deepcopy(self.errorState);
        stateAfterEnvChange = copy.deepcopy(self.errorState);
        measAfterAction = [-2, -1000, -1000]
        self.net.switch.at[0, 'closed'] = True
        self.net.switch.at[1, 'closed'] = False
        if lp_ref != 'na' and v_ref_pu != 'na':
            self.shuntControl.ref = v_ref_pu;
            self.seriesControl.ref = lp_ref;
        networkFailure = False
        done = False;
        bus_index_shunt = 1;
        line_index = 1;
        if self.stateIndex < min(len(self.powerProfile), len(self.loadProfile)):
            dummyRes = (self.net.res_bus.vm_pu, self.net.res_line.loading_percent)
            ## state = (voltage,ll,angle,p,q)
            pp.runpp(self.net, run_control=True);
            if self.method in ('dqn', 'ddqn', 'td3'):
                reward1 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent,
                                               self.net.res_bus.va_degree[bus_index_shunt]);
                stateAfterAction = self.getCurrentStateForDQN()
            else:
                reward1 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
                stateAfterAction = self.getCurrentState()
            # print('rew1: ', reward1)
            measAfterAction = [self.net.res_bus.vm_pu[1], max(self.net.res_line.loading_percent),
                               np.std(self.net.res_line.loading_percent)]
            done = self.stateIndex == (len(self.powerProfile) - 1)
            if done == False:
                self.incrementLoadProfile()
                if self.method in ('dqn', 'ddqn', 'td3'):
                    reward2 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent,
                                                   self.net.res_bus.va_degree[bus_index_shunt]);
                    stateAfterEnvChange = self.getCurrentStateForDQN()
                else:
                    reward2 = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
                    stateAfterEnvChange = self.getCurrentState()
            # print('rew2: ',reward2)
            reward = 0.7 * reward1 + 0.3 * reward2;
        else:
            print('wrong block!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        stateAfterEnvChange.extend(stateAfterAction)
        # print(self.errorState)

        # print(reward2)
        # print('totrew: ', reward)
        return stateAfterEnvChange, reward, done or networkFailure, measAfterAction;

    def incrementLoadProfile(self):
        self.stateIndex += 1;
        self.scaleLoadAndPowerValue(self.stateIndex);
        self.runEnv(True);

        """
        try:
            pp.runpp(self.net);
            reward = self.calculateReward(self.net.res_bus.vm_pu, self.net.res_line.loading_percent);
        except:
            networkFailure=True;
            self.net.shunt.q_mvar=shuntBackup;
            self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']]=impedenceBackup;
            pp.runpp(self.net);
            reward=1000;
            return self.net.res_bus,reward,True;
        self.stateIndex += 1;
        if self.stateIndex < len(self.powerProfile):
            if (self.scaleLoadAndPowerValue(self.stateIndex, self.stateIndex - 1) == False):
                networkFailure = True;
                reward = 1000;
                # self.stateIndex -= 1;
        return self.net.res_bus, reward, self.stateIndex == len(self.powerProfile) or networkFailure;
        """

    ##Function to calculate line reactance in pu
    def X_pu(self, line_index):
        s_base = 100e6
        v_base = 230e3
        x_base = pow(v_base, 2) / s_base
        x_line_ohm = self.net.line.x_ohm_per_km[line_index]
        x_line_pu = x_line_ohm / x_base  # Can take one since this line is divivded into
        # 2 identical lines with length 0.5 km
        #print(x_line_pu)
        return x_line_pu

    ## Resets environment choosing new starting state, used for beginning of each episode
    def reset(self):
        self.stateIndex = self.getstartingIndex()

        #Disable FACTS
        self.net.switch.at[0, 'closed'] = False
        self.net.switch.at[1, 'closed'] = True

        # Make sure FACTS output is reset for controllers to work properly
        #print(self.net.shunt.q_mvar[0])
        #self.net.shunt.q_mvar[0] = 0
        #print(self.net.impedance.loc[0, ['xft_pu']])
        #self.net.impedance.loc[0, ['xft_pu', 'xtf_pu']] =
        #self.net.shunt.q_mvar

        self.scaleLoadAndPowerValue(self.stateIndex);
        try:
            pp.runpp(self.net, run_control=False);
        except:
            print('Some error occurred while resetting the environment');
            raise Exception('cannot proceed at these settings. Please fix the environment settings');

    ## Calculate immediate reward
    def calculateReward(self, voltages, loadingPercent,loadAngle=10):
        try:
            rew=0;
            for i in range(1,2):
                if voltages[i]  > 1:
                    rew=voltages[i]-1;
                else:
                    rew=1-voltages[i];
                rewtemp = rew # For storage to set reward to 0
            rew = math.exp(rew*10)*-20;
            #print(rew)
            loadingPercentInstability=np.std(loadingPercent)# Think it works better without this addition: * len(loadingPercent);
            rew = rew - loadingPercentInstability;
            # (math.exp(abs(1-voltages[i])*10)*-20)-std ;
            #print(rew)
            #rew=rew if abs(loadAngle)<30 else rew-200;
        except:
            print('exception in calculate reward')
            print(voltages);
            print(loadingPercent)
            return 0;
        rew = (200+rew)/200 # normalise between 0-1
        if rewtemp > 0.15 or abs(loadAngle)>=30: # IF voltage deviating more than 0.15 pu action is very very bad.
            rew = 0.001 #Also makes sure that final rew >=0
        if rew < 0:
            rew = 0
        return rew

    ## Simple plot diagram
    def plotGridFlow(self):
        print('plotting powerflow for the current state')
        plot.simple_plot(self.net)

    ## Scale load and generation from load and generation profiles
    def scaleLoadAndPowerValue(self,index):

        scalingFactorLoad = self.loadProfile[index] / (sum(self.loadProfile)/len(self.loadProfile));
        scalingFactorPower = self.powerProfile[index] / max(self.powerProfile);

        self.net.load.p_mw[0] = self.nominalP * scalingFactorLoad;
        self.net.load.q_mvar[0] = self.nominalQ * scalingFactorLoad;
        #self.net.sgen.p_mw = self.net.sgen.p_mw * scalingFactorPower;
        #self.net.sgen.q_mvar = self.net.sgen.q_mvar * scalingFactorPower;

    def runNoFACTS(self, busVoltageInd):
        # Bypass FACTS devices if wantd
        self.net.switch.at[0, 'closed'] = True
        self.net.switch.at[1, 'closed'] = False
        self.net.controller.in_service[0] = True
        self.net.controller.in_service[1] = True
        self.shuntControl.ref = 1
        self.seriesControl.ref = 50

        # Create array
        v_arr = []
        l_arr = []

        # Loop through all loadings
        for i in range(0, 600): #len(self.loadProfile)
            # Increment and run environment
            self.stateIndex += 1;
            self.scaleLoadAndPowerValue(self.stateIndex);
            self.runEnv(True);

            # Store result for current settings
            v_arr.append(self.net.res_bus.vm_pu[busVoltageInd])
            l_arr.append(self.stateIndex)

        # Plot result
        print(max(v_arr))
        print(min(v_arr))
        plt.plot(l_arr, v_arr)
        plt.grid()
        plt.xlabel('Time step on load profile [-]', fontsize= 18 )
        plt.ylabel('Voltage [pu]', fontsize= 18)
        plt.title('Bus 2 Voltage with shunt+series FACTS ', fontsize= 22)
        plt.show()

    def runNoRL(self, busVoltageInd):
        # Print the load profile:
        # loadProfilesScaled = self.loadProfile / (sum(self.loadProfile) / len(self.loadProfile))
        # P = loadProfilesScaled * self.nominalP
        # Q = loadProfilesScaled * self.nominalQ
        # xaxis = range(0, len(self.loadProfile))
        # fig, ax1 = plt.subplots()
        # ax1.set_title('Load profile', fontsize=24)
        # ax1.set_xlabel('Time step on load profile [-]', fontsize=20)
        # ax1.set_ylabel('Active power [MW] ', color='r', fontsize=20)
        # ax1.plot(xaxis, P, color='r')
        # ax1.set_ylim(0, 500)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Reactive power [Mvar] ', color='tab:blue', fontsize=20)
        # ax2.plot(xaxis, Q, color='tab:blue')
        # ax2.set_ylim(0,500)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.grid()
        # plt.show()
        #
        # #Zoomed in version:
        # fig, ax1 = plt.subplots()
        # ending = 1000-1
        # ax1.set_title('Load profile', fontsize=24)
        # ax1.set_xlabel('Time step on load profile [-]', fontsize=20)
        # ax1.set_ylabel('Active power [MW] ', color='r', fontsize=20)
        # ax1.plot(xaxis[0:ending], P[0:ending], color='r')
        # ax1.set_ylim(0,500)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Reactive power [Mvar] ', color='tab:blue', fontsize=20)
        # ax2.plot(xaxis[0:ending], Q[0:ending], color='tab:blue')
        # ax2.set_ylim(0,500)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.grid()
        # plt.show()

        #SHUNT+SERIES:
        # Bypass FACTS devices if wantd
        self.net.switch.at[0, 'closed'] = True
        self.net.switch.at[1, 'closed'] = True
        self.net.controller.in_service[0] = True
        self.net.controller.in_service[1] = False
        self.shuntControl.ref = 1
        self.seriesControl.ref = 50

        # Create array
        v_arr = []
        v_arr_so = []
        l_arr = []

        # Loop through all loadings
        for i in range(0, 600):  # len(self.loadProfile)
        # Increment and run environment
            self.stateIndex += 1;
            self.scaleLoadAndPowerValue(self.stateIndex);
            self.runEnv(True);
            # Store result for current settings
            v_arr_so.append(self.net.res_bus.vm_pu[busVoltageInd])
            l_arr.append(self.stateIndex)

        #SHUNT ONLY
        self.setMode('test')
        self.net.switch.at[0, 'closed'] = True
        self.net.switch.at[1, 'closed'] = False
        self.net.controller.in_service[0] = True
        self.net.controller.in_service[1] = True

        for i in range(0, 600):  # len(self.loadProfile)
        # Increment and run environment
            self.stateIndex += 1;
            self.scaleLoadAndPowerValue(self.stateIndex);
            self.runEnv(True);
            # Store result for current settings
            v_arr.append(self.net.res_bus.vm_pu[busVoltageInd])

        # Plot result
        print(max(v_arr))
        print(min(v_arr))
        print(max(v_arr_so))
        print(min(v_arr_so))
        plt.plot(l_arr, v_arr)
        plt.plot(l_arr, v_arr_so)
        plt.grid()
        plt.xlabel('Time step on load profile [-]', fontsize=20)
        plt.ylabel('Voltage [pu]', fontsize=20)
        plt.title('Bus 2 Voltage with non-RL FACTS ', fontsize=24)
        plt.legend(['shunt+series','shunt only'], fontsize=12)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()



##Load Profile data has been pickled already, do not run this function for now
def createLoadProfile():
    ML  = (np.cos(2 * np.pi/12 * np.linspace(0,11,12)) * 50 + 100 ) * 1000  # monthly load
    ML = el.make_timeseries(ML) #convenience wrapper around pd.DataFrame with pd.DateTimeindex
    #print(ML)
    DWL =  el.gen_daily_stoch_el() #daily load working
    DNWL = el.gen_daily_stoch_el() #daily load non working
    #print(sum(DNWL))
    Weight = .60 # i.e energy will be split 55% in working day 45% non working day

    Load1 =  el.gen_load_from_daily_monthly(ML, DWL, DNWL, Weight)
    Load1.name = 'L1'
    Load1=Load1.round();
    #print(Load1)

    disag_profile = np.random.rand(60)
    JanLoadEveryMinute=el.generate.disag_upsample(Load1[0:744],disag_profile, to_offset='min');
    JanLoadEvery5mins=[];
    l=0;
    for i in range(0,JanLoadEveryMinute.shape[0]):
        l=l+JanLoadEveryMinute[i];
        if np.mod(i+1,5) == 0:
            JanLoadEvery5mins.append(l);
            l=0;

    windDataDF = pd.read_excel('Data/WindEnergyData.xlsx');
    generatorValuesEvery5mins=[];
    for i in range(1,windDataDF.shape[0]):
        randomValue=np.random.choice(100, 1)[0]
        randomValue_prob = np.random.random();
        if randomValue > windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'] or randomValue_prob < 0.4:
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'])
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'])
        else :
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'] - randomValue)
            generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'] + randomValue)
        generatorValuesEvery5mins.append(windDataDF.iloc[i]['DE_50hertz_wind_generation_actual'])
    print(len(generatorValuesEvery5mins))
    print(len(JanLoadEvery5mins))
    pickle.dump(generatorValuesEvery5mins, open("Data/generatorValuesEvery5mins.pkl", "wb"))
    pickle.dump(JanLoadEvery5mins, open("Data/JanLoadEvery5mins.pkl", "wb"))



def trainTestSplit():
    with open('Data/JanLoadEvery5mins.pkl', 'rb') as pickle_file:
        loadProfile = pickle.load(pickle_file)
    numOFTrainingIndices =  int(np.round(0.8*len(loadProfile)))
    trainIndices=np.random.choice(range(0,len(loadProfile)),numOFTrainingIndices,replace=False)
    trainIndicesSet=set(trainIndices)
    testIndices=[x for x in range(0,len(loadProfile)) if x not in trainIndicesSet]
    pickle.dump(trainIndices, open("Data/trainIndices.pkl", "wb"))
    pickle.dump(testIndices, open("Data/testIndices.pkl", "wb"))
    #print(len(loadProfile))
    #print(len(trainIndicesSet))
    #print(len(trainIndices))
    #print(len(testIndices))

#createLoadProfile()
#trainTestSplit()



# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self):
        self.n_actions = len(act)
        self.n_features = n_f
        self.n_neurons_hidden_layer = n_n
        self.lr = l_r
        self.gamma = r_w
        self.replace_target_iter = r_t_i
        self.memory_size = m_s
        self.batch_size = b_s
        self.epsilon_increment = e_g_enc
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon = epsil
        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))  # 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # consist of [target_net, evaluate_net]
        self._build_net()  # build net arfter init

        with tf.variable_scope('soft_replacement'):  #
            self.target_with_evolve_replace = [tf.assign(t, e) for t, e in
                                               zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                     scope='target_net'),
                                                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                     scope='eval_net'))]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        if output_graph:
            tf.summary.FileWriter("C:/tensorboard/DQN/", self.sess.graph)
        self.cost_his = []

    def decrease_greedy(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < 1.0 else 1.0
        self.learn_step_counter += 1

    def decrease_learning_rate(self):
        self.lr = self.lr * 0.99

    def _build_net(self):

        ''' __________________ inputs _________________________'''

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # State / count of input neurons
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        weights = tf.random_normal_initializer(0., 0.3)

        if bias_on == True:
            bias = tf.constant_initializer(0.1)
            ''' ____________________________ evaluate_net ____________________________________'''
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, units=self.n_neurons_hidden_layer, activation=tf.nn.relu,
                                     kernel_initializer=weights, bias_initializer=bias, name='e1')
                self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=weights, bias_initializer=bias,
                                              name='q')
            ''' ______________________________ target_net ______________________________________'''
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, units=self.n_neurons_hidden_layer, activation=tf.nn.relu,
                                     kernel_initializer=weights, bias_initializer=bias, name='t1')
                self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=weights, bias_initializer=bias,
                                              name='t2')
        else:
            ''' ____________________________ evaluate_net ____________________________________'''
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, units=self.n_neurons_hidden_layer, activation=tf.nn.relu,
                                     kernel_initializer=weights, name='e1')
                self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=weights, name='q')
            ''' ______________________________ target_net ______________________________________'''
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, units=self.n_neurons_hidden_layer, activation=tf.nn.relu,
                                     kernel_initializer=weights, name='t1')
                self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=weights, name='t2')

        # Bellman's equation: Q(s, a) = r(s, a) + yQ(s', a)
        # s - current state (q_eval), s' - next state (q_next)
        # Q(s, a) calculating by target net:
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1,
                                                           name='Qmax_s_')  # shape=(None, ) reduce to vector with shape = axis
            # q_target = self.r + self.gamma * np.amax(self.q_next, axis = 1) np.max or tf.reduce_max
            self.q_target = tf.stop_gradient(
                q_target)  # no calculating gradient in this net (wheights improwed by "self.target_with_evolve_replace")

        # q_eval give us the actions of neural net in current state (s):
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )

        with tf.variable_scope('loss'):
            # self.loss =S tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))  # origina
            self.loss = tf.reduce_mean((self.q_target - self.q_eval_wrt_a) ** 2)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            # self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            # self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            # self._train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss)
            # self._train_op =tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
            # self._train_op =tf.train.FtrlOptimizer(self.lr).minimize(self.loss)
            # self._train_op =tf.train.MomentumOptimizer(self.lr).minimize(self.loss)
            # self._train_op =tf.train.ProximalAdagradOptimizer(self.lr).minimize(self.loss)
            # self._train_op =tf.train.ProximalGradientDescentOptimizer(self.lr).minimize(self.loss)

        # print(self.loss)

    def store_transition(self, s, a, r, s_):  # replay memory
        if not hasattr(self, 'memory_counter'):  #
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        # print('stored')

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]  # to have batch dimension when feed into tf placeholder
        if np.random.uniform() < self.epsilon:
            # feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(self.n_actions)
        # print('e-greedy = ', self.epsilon)
        # print('action',action)
        return action

    '''

     # old 
    def _replace_target_params(self):
        targNET_param = tf.get_collection('taregt_net_params')
        evalNET_param = tf.get_collection('taregt_net_params')
        self.sess.run([tf.assign(t,e) for t, e in zip{targNET_param, evalNET_param}])
    '''

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:  # replace target net after # iterations
            self.sess.run(self.target_with_evolve_replace)
            # print(self.learn_step_counter, '\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: batch_memory[:, :self.n_features],
                                                                        self.a: batch_memory[:, self.n_features],
                                                                        self.r: batch_memory[:, self.n_features + 1],
                                                                        self.s_: batch_memory[:, -self.n_features:], })

        self.cost_his.append(cost)
        self.learn_step_counter += 1  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def run_env():
    step = 0

    for episode in range(1):  #
        # initial observation
        observation = env.reset()

        while step <= 5000:
            # while True:
            # refresh
            env.render()

            if random_consumption_on == True:
                rand = random.random()
                if (rand < 0.1):
                    env.increase_consumption()
                elif (rand > 0.9):
                    env.decrease_consumption()

            # RL choose action based on observation

            action = RL.choose_action(observation)
            # RL take action and get next observation/reward
            observation_, reward = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 100 and step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # if (step > 1000 and step % 100 == 0):
            #   RL.decrease_learning_rate()

            if (step > start_increase_e_greedy and step % 100 == 0):
                RL.decrease_greedy()

            file1 = open(r"./DQN_step.txt", "a")
            file1.write(str(step) + ';')
            file1.close
            file2 = open(r"./DQN_freq.txt", "a")
            file2.write(str(env.frequency) + ';')
            file2.close

            step += 1
            # print('step: ', step)

        # print weidhts
        variables = tf.trainable_variables()
        print(variables)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        print("Weight matrix: {0}".format(sess.run(variables[2])))

        import matplotlib.pyplot as plt
        import numpy as np
        import math

        a = open('./DQN_step.txt', 'r')
        b = a.read().split(';')
        b.pop()
        x = [float(i) for i in b]
        c = open('./DQN_freq.txt', 'r')
        d = c.read().split(';')
        d.pop()
        y = [float(i) for i in d]
        fig, ax = plt.subplots()
        ax.plot(x, y, color="red", label="DQN")
        ax.set_xlabel("step")
        ax.set_ylabel("frequency")
        ax.legend()
        fig.savefig('./DQN.png')


if __name__ == "__main__":
    env = powerGrid_ieee4()
    RL = DeepQNetwork()
    env.after(100, run_env())
    env.mainloop()