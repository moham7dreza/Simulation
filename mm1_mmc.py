__metaclass__ = type

import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from queue import Queue
import math


class Task(object):

    def __init__(self, simulation_time):
        self.simulation_time = simulation_time

    def get_time(self):
        return self.simulation_time

    def __repr__(self):
        return "Task created at %s" % self.simulation_time


class MM1(object):

    def __init__(self):
        self.IDLE = 0
        self.BUSY = 1
        self.DEPARTURE_REFERENCE = random.getrandbits(128)
        # print(self.DEPARTURE_REFERENCE)
        self.state = self.IDLE
        self.state_audit = []
        self._queue = Queue()

    def plot_result_array(self, array_to_plot, figure, title="Unnamed", xlabel="X", ylabel="Y", block=False):
        plt.figure(figure)
        n, bins, patches = plt.hist(array_to_plot, 50, density=1, facecolor='green', alpha=0.75)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show(block=block)

    def plot_queue_audit(self, queue_to_plot, figure, title="Unamed", xlabel="X", ylabel="Y", block=False):
        time_values = [time for (time, qsize) in queue_to_plot]
        qsize_values = [qsize for (time, qsize) in queue_to_plot]
        plt.figure(figure)
        plt.plot(time_values, qsize_values)
        plt.grid(True)
        plt.show(block=block)

    def set_state(self, state):
        self.state = state
        self.state_audit.append(state)

    def update_statistics(self):
        self.time_since_last_event = self.sim_time - self.last_event_time
        self.cumulated_queue_len += self._queue.qsize() * self.time_since_last_event

        if self.state == self.BUSY:
            self.total_busy_time += self.time_since_last_event

    def mm1(self, _lambda, _mi, max_requests):
        Ntotal = 0
        ntot = []
        n = 0
        nq = []
        simulation_start_time = 0
        self.sim_size = 0
        self.sim_time = 0
        self.cumulated_queue_len = 0
        self.last_event_time = 0
        self.total_busy_time = 0
        queue_audit = []
        next_departure = simulation_start_time
        next_arrival = np.random.exponential(1 / _lambda)

        while self.sim_size < max_requests:
            if next_arrival < next_departure:
                self.sim_size += 1
                self.sim_time = next_arrival
                self.update_statistics()
                queue_audit.append((self.sim_time, self._queue.qsize()))
                if self.state == self.IDLE:
                    self.set_state(self.BUSY)
                    next_departure = self.sim_time + np.random.exponential(1 / _mi)
                    n += 1
                else:
                    new_task = Task(self.sim_time)
                    self._queue.put(new_task)
                queue_audit.append((self.sim_time, self._queue.qsize()))
                next_arrival = self.sim_time + np.random.exponential(1 / _lambda)
            else:
                self.sim_time = next_departure
                self.update_statistics()
                if self._queue.empty():
                    self.set_state(self.IDLE)
                    next_departure = self.DEPARTURE_REFERENCE
                else:
                    self._queue.get()
                    next_departure = self.sim_time + np.random.exponential(1 / _mi)
                    queue_audit.append((self.sim_time, self._queue.qsize() + 1))
            self.last_event_time = self.sim_time
            queue_audit.append((self.sim_time, self._queue.qsize()))
            nq.append(self._queue.qsize())
            ntot.append(self._queue.qsize() + n)

        # Simulation total time
        simulation_end_time = self.last_event_time
        simulation_total_time = simulation_end_time - simulation_start_time

        avg_queue_length = self.cumulated_queue_len / self.sim_time
        avg_utilization = self.total_busy_time / self.sim_time

        # output to the screen list of values
        # print("Confidence interval for Nw")
        # lq, uq = calculate_Confidence_interval(nq)
        # print("lower bound : " + str(lq) + "  upper bound : " + str(uq))
        print("Utilization : ", avg_utilization)
        print("Idle : ", 1 - avg_utilization)
        print("simulation_total_time : " + str(simulation_total_time))
        print("avg queue length Nw : " + str(avg_queue_length))
        return simulation_total_time, avg_queue_length, avg_utilization, nq

    # define a class called 'Customer'


# objects of the customer
class Customer:
    def __init__(self, arrival_time, service_start_time, service_time):
        self.arrival_time = arrival_time
        self.service_start_time = service_start_time
        self.service_time = service_time
        self.service_end_time = self.service_start_time + self.service_time
        self.wait = self.service_start_time - self.arrival_time


# Average Utilization p
def P(arrivalRate, serviceRate):
    return float(arrivalRate / serviceRate) * 100


# Factor or idle percentage of the system
def POcio(arrivalRate, serviceRate):
    return 100 - P(arrivalRate, serviceRate)


# Expected number of clients in the queue (Lq)
def Lq(arrivalRate, serviceRate):
    return float((arrivalRate * arrivalRate) / (serviceRate * (serviceRate - arrivalRate)))


# Expected number of clients receiving the service (Ls)
def Ls(arrivalRate, serviceRate):
    return arrivalRate / serviceRate


# Expected number of clients in the queue system (Lw)
def Lw(arrivalRate, serviceRate):
    return arrivalRate / (serviceRate - arrivalRate)


# Expected value of time spent by a client in the queue (Wq)
def Wq(arrivalRate, serviceRate):
    return arrivalRate / (serviceRate * (serviceRate - arrivalRate))


# Expected value of the time a customer spends on the service (Ws)
def Ws(arrivalRate, serviceRate):
    return 1 / serviceRate


# Expected value of the time a client takes to traverse the system (Ww)
def Ww(arrivalRate, serviceRate):
    return 1 / (serviceRate - arrivalRate)


def time_based(lambd, mu, simulation_time):
    delta_t = 1
    t = 0
    N_s = 0
    N_q = 0
    while t < simulation_time:
        t += delta_t
        if draw(lambd * delta_t):
            N_q += 1
        if N_s == 1 and draw(mu * delta_t) and N_q > 0:
            N_q -= 1
        else:
            N_s = 0
        if N_s == 0 and N_q > 0:
            N_s = 1
            N_q -= 1
        print(t, N_q, N_s)


def draw(p):
    rand = np.random.uniform(0, 1)
    if rand < p:
        return True
    else:
        return False


def event_based(lambd, mu, simulation_time):
    t = 0
    N_s = 0
    N_q = 0
    while t < simulation_time:
        if N_s == 1:
            next_arrival = np.random.exponential(1 / lambd)
            next_departure = np.random.exponential(1 / mu)
            if next_arrival > next_departure:
                t += next_departure
                if N_q > 0:
                    N_q -= 1
                else:
                    N_s = 0
            else:
                t += next_arrival
                N_q += 1
        else:
            next_arrival = np.random.exponential(1 / lambd)
            t += next_arrival
            N_s = 1
        print(t, N_q, N_s)


# a simple function to sample from negative exponential
def generate_expo(lambd):
    return random.expovariate(lambd)


def calculate_Confidence_interval(arr):
    z_90_percent = 1.645
    z_95_percent = 1.960
    z_99_percent = 2.576
    mean = np.mean(arr)
    var = np.var(arr)
    lower_bound = mean - z_99_percent * (var / np.sqrt(len(arr)))
    upper_bound = mean + z_99_percent * (var / np.sqrt(len(arr)))
    return lower_bound, upper_bound


# function callback lambda, mu, simulation_time
def foo(lambd, mu, simulation_time):
    Nc = 0
    # initialise time to 0
    t = 0
    while t < simulation_time:

        # calculate arrival time and service time for new customer
        if len(Customers) == 0:
            arrival_time = np.random.exponential(1 / lambd)
            service_start_time = arrival_time
        # added arrival time
        else:
            arrival_time += np.random.exponential(1 / lambd)
            service_start_time = max(arrival_time, Customers[-1].service_end_time)
            # print(str(Customers[-1].service_end_time))
        service_time = np.random.exponential(1 / mu)
        # create new customer
        Customers.append(Customer(arrival_time, service_start_time, service_time))
        Nc += 1
        # increment clock till next end of service
        t = arrival_time

    # ----------------------------------

    # calculate summary statistics
    Waits = [a.wait for a in Customers]
    Mean_Wait = sum(Waits) / len(Waits)

    Total_Times = [a.wait + a.service_time for a in Customers]
    Mean_Time = sum(Total_Times) / len(Total_Times)

    Service_Times = [a.service_time for a in Customers]
    Mean_Service_Time = sum(Service_Times) / len(Service_Times)

    # Server Utilisation
    # Utilisation = sum(Service_Times) / t

    # output to the screen list of values
    print("[ + ] Simulation Results ...")
    print("Arrival rate: ", lambd)
    print("Number of customers: ", len(Customers))
    print("Mean Service Time: ", Mean_Service_Time)
    print("Mean Wait Time: ", Mean_Wait)
    print("Observed Response time in the system: ", Mean_Time)
    # print("Utilization : ", Utilisation)
    # print("Idle : ", 1 - Utilisation)
    print("Theoretical Result Calculated", Theoretical_result)
    # print("Confidence level : 99%")
    # print("Confidence interval for W")
    # lw, uw = calculate_Confidence_interval(Waits)
    # print("lower bound : " + str(lw) + " upper bound : " + str(uw))
    # print("Confidence interval for R")
    # lr, ur = calculate_Confidence_interval(Total_Times)
    # print("lower bound : " + str(lr) + " upper bound : " + str(ur))
    # print("Confidence interval for S")
    # ls, us = calculate_Confidence_interval(Service_Times)
    # print("lower bound : " + str(ls) + " upper bound : " + str(us))
    # output to the file to store customer entire enter and exit time
    outfile2 = open('main4/MM1Q-detailed_log_output-(%s,%s,%s).csv' % (lambd, mu, simulation_time), 'w')
    output2 = csv.writer(outfile2)
    output2.writerow(['Customer', 'Arrival_time', 'Wait', 'Service_Start_time', 'Service_Time', 'Service_End_time'])
    f = 0
    for customer in Customers:
        f = f + 1
        outrow2 = []
        outrow2.append(f)
        outrow2.append(customer.arrival_time)
        outrow2.append(customer.wait)
        outrow2.append(customer.service_start_time)
        outrow2.append(customer.service_time)
        outrow2.append(customer.service_end_time)
        output2.writerow(outrow2)
    outfile2.close()
    return Mean_Time, Mean_Wait, Mean_Service_Time, Waits, Total_Times, Service_Times


class mmc:
    def __init__(self):
        self.num_in_system = 0
        self.clock = 0.0  # simulation clock
        self.num_arrivals = 0  # total number of arrivals
        self.t_arrival = np.random.exponential(1 / lambd)  # time of next arrival
        self.t_departure1 = float('inf')  # departure time from server 1
        self.t_departure2 = float('inf')  # departure time from server 2
        self.dep_sum1 = 0  # Sum of service times by server 1
        self.dep_sum2 = 0  # Sum of service times by server 2
        self.state_T1 = 0  # current state of server1 (binary)
        self.state_T2 = 0  # current state of server2 (binary)
        self.total_wait_time = 0.0  # total wait time
        self.num_in_q = 0  # current number in queue
        self.number_in_queue = 0  # customers who had to wait in line(counter)
        self.num_of_departures1 = 0  # number of customers served by teller 1
        self.num_of_departures2 = 0  # number of customers served by teller 2

    def mmc(self, lam, mu, simulation_time):
        print(str(lam) + " " + str(mu))
        while self.clock < simulation_time:
            t_next_event = min(self.t_arrival, self.t_departure1, self.t_departure2)
            self.total_wait_time += (self.num_in_q * (t_next_event - self.clock))
            self.clock = t_next_event

            # arrival
            if self.t_arrival < self.t_departure1 and self.t_arrival < self.t_departure2:
                self.num_arrivals += 1
                self.num_in_system += 1

                if self.num_in_q == 0:  # schedule next departure or arrival depending on state of servers
                    if self.state_T1 == 1 and self.state_T2 == 1:
                        self.num_in_q += 1
                        self.number_in_queue += 1
                        self.t_arrival = self.clock + np.random.exponential(1 / lam)
                    elif self.state_T1 == 0 and self.state_T2 == 0:
                        if np.random.choice([0, 1]) == 1:
                            self.state_T1 = 1
                            self.dep1 = np.random.exponential(1 / mu)
                            self.dep_sum1 += self.dep1
                            self.t_departure1 = self.clock + self.dep1
                            self.t_arrival = self.clock + np.random.exponential(1 / lam)
                        else:
                            self.state_T2 = 1
                            self.dep2 = np.random.exponential(1 / mu)
                            self.dep_sum2 += self.dep2
                            self.t_departure2 = self.clock + self.dep2
                            self.t_arrival = self.clock + np.random.exponential(1 / lam)
                    elif self.state_T1 == 0 and self.state_T2 == 1:  # if server 2 is busy customer goes to server 1
                        self.dep1 = np.random.exponential(1 / mu)
                        self.dep_sum1 += self.dep1
                        self.t_departure1 = self.clock + self.dep1
                        self.t_arrival = self.clock + np.random.exponential(1 / lam)
                        self.state_T1 = 1
                    else:  # otherwise customer goes to server 2
                        self.dep2 = np.random.exponential(1 / mu)
                        self.dep_sum2 += self.dep2
                        self.t_departure2 = self.clock + self.dep2
                        self.t_arrival = self.clock + np.random.exponential(1 / lam)
                        self.state_T2 = 1
                else:
                    self.num_in_q += 1
                    self.number_in_queue += 1
                    self.t_arrival = self.clock + np.random.exponential(1 / lam)
            # server 1
            elif self.t_departure1 < self.t_arrival and self.t_departure1 < self.t_departure2:
                self.num_of_departures1 += 1
                if self.num_in_q > 0:
                    self.dep1 = np.random.exponential(1 / mu)
                    self.dep_sum1 += self.dep1
                    self.t_departure1 = self.clock + self.dep1
                    self.num_in_q -= 1
                else:
                    self.t_departure1 = float('inf')
                    self.state_T1 = 0
            # server 2
            else:
                self.num_of_departures2 += 1
                if self.num_in_q > 0:
                    self.dep2 = np.random.exponential(1 / mu)
                    self.dep_sum2 += self.dep2
                    self.t_departure2 = self.clock + self.dep2
                    self.num_in_q -= 1
                else:
                    self.t_departure2 = float('inf')
                    self.state_T2 = 0


def Erlang_C_Formula(rho, c):
    numerator = (math.pow(rho * c, c) / math.factorial(c)) * (1 / (1 - rho))
    summation = 0
    for n in range(0, c - 1):
        summation += math.pow(rho * c, n) / math.factorial(n)
    denominator = summation + numerator
    result = numerator / denominator
    return result


def get_exp(rate):
    uniform = np.random.uniform(0, 1)
    ln_uniform = - np.log(1 - uniform)
    result = ln_uniform / rate
    return result


if __name__ == '__main__':
    # main function
    # instance variables
    lambd = 0.1
    mu = 1
    simulation_time = 1000
    i = 0
    Theoretical_result = 0.0
    # compare result with obtained and store in file to experiment
    outfile1 = open('main4/MM1.csv', 'w')
    output1 = csv.writer(outfile1)
    output1.writerow(['No', 'Arrival_Rate', 'Observed Response time', 'Theoretical_result'])

    rho_stored = []
    rho2_stored = []

    R_stored = []
    LR_stored = []
    UR_stored = []
    RR_stored = []

    WW_stored = []
    W_stored = []
    LW_stored = []
    UW_stored = []

    SS_stored = []
    S_stored = []
    LS_stored = []
    US_stored = []

    N_stored = []
    NN_stored = []

    U_stored = []
    UU_stored = []
    LowU_stored = []
    upU_stored = []

    LNw_stored = []
    UNw_stored = []
    Nw_stored = []
    Nww_stored = []

    R2_stored = []
    LR2_stored = []
    UR2_stored = []
    RR2_stored = []

    WW2_stored = []
    W2_stored = []
    LW2_stored = []
    UW2_stored = []

    SS2_stored = []
    S2_stored = []
    LS2_stored = []
    US2_stored = []

    U2_stored = []
    UU2_stored = []
    LowU2_stored = []
    upU2_stored = []

    LNw2_stored = []
    UNw2_stored = []
    Nw2_stored = []
    Nww2_stored = []

    df = pd.DataFrame(
        columns=['Arrival Rate',
                 'Average interarrival time',
                 'Average service time server1',
                 'Average service time server 2',
                 'Utilization server 1',
                 'Utilization server 2',
                 'People who had to wait in line',
                 'Total average wait time',
                 'Response time',
                 'Total Utilization'])
    df2 = pd.DataFrame(
        columns=['lambda',
                 'mu',
                 'Rho',
                 'Utilization(Sim)',
                 'CI lower bound for U',
                 'CI upper bound for U',
                 'S',
                 'Service Time(Sim)',
                 'CI lower bound for S',
                 'CI upper bound for S',
                 'W',
                 'Waiting Time(Sim)',
                 'CI lower bound for W',
                 'CI upper bound for W',
                 'R',
                 'Response Time(Sim)',
                 'CI lower bound for R',
                 'CI upper bound for R',
                 'Nw',
                 'Queue Length(Sim)',
                 'CI lower bound for Nw',
                 'CI upper bound for Nw',
                 'Simulation Total Time',
                 ])
    df3 = pd.DataFrame(
        columns=['lambda',
                 'mu',
                 'arr',
                 'dep 1',
                 'dep 2',
                 'Rho',
                 'Utilization(Sim)',
                 'CI lower bound for U',
                 'CI upper bound for U',
                 'S',
                 'Service Time(Sim)',
                 'CI lower bound for S',
                 'CI upper bound for S',
                 'W',
                 'Waiting Time(Sim)',
                 'CI lower bound for W',
                 'CI upper bound for W',
                 'R',
                 'Response Time(Sim)',
                 'CI lower bound for R',
                 'CI upper bound for R',
                 'Nw',
                 'Queue Length(Sim)',
                 'CI lower bound for Nw',
                 'CI upper bound for Nw',
                 ])

    # generate ten arrival rates starting from 0.1 till 0.9
    for x in range(1, 20):

        print("*********************************************************")
        print("[ + ] Analytical Results mm1")
        rho = lambd / mu
        S = 1 / mu
        N = rho / (1 - rho)
        T = N / lambd
        W = T * rho
        N_w = N * rho  # lambd * W
        print("lambda : ", str(lambd))
        print("Mu : ", str(mu))
        print("E[s] avg service time : ", str(S))
        print("U utilization: ", str(rho))
        print("N avg numbers in system : ", str(N))
        # print("N2 avg numbers in system : ", str(lambd * (W + S)))
        print("T avg response time : ", str(T))
        # print("T2 avg response time : ", str(W + S))
        print("W avg waiting time : ", str(W))
        print("N_w  avg numbers in waiting queue: ", str(N_w))
        # print("N_w2  avg numbers in waiting queue: ", str(lambd * W))

        print("*********************//////***")
        print("[ + ] Analytical Results mmc")
        c = 2
        rho2 = lambd / (mu * c)
        S2 = 1 / mu
        N2 = ((rho2 / (1 - rho2)) * Erlang_C_Formula(rho, c)) + (c * rho2)
        T2 = (Erlang_C_Formula(rho, c) / ((c * mu) - lambd)) + (1 / mu)
        W2 = Erlang_C_Formula(rho, c) / c * (1 - rho2) * mu
        N_w2 = lambd * W2
        print("lambda : ", str(lambd))
        print("Mu : ", str(mu))
        print("E[s] avg service time : ", str(S2))
        print("U utilization: ", str(rho2))
        print("N avg numbers in system : ", str(N2))
        print("T avg response time : ", str(T2))
        print("W avg waiting time : ", str(W2))
        print("N_w  avg numbers in waiting queue: ", str(N_w2))

        Theoretical_result = 1 / (mu - lambd)
        Customers = []
        # function call
        # Mean_Time, lr, ur, Mean_Wait, lw, uw, Mean_Service_Time, ls, us, len_Customers = foo(lambd, mu, simulation_time)
        # mm1 = MM1()
        # simulation_total_time, avg_queue_length, avg_utilization, lq, uq = mm1.run_simulation(lambd, mu, 1000)
        ResponseTimes = []
        WaitingTimes = []
        ServiceTimes = []
        Qlens = []
        Utils = []

        ResponseTimes2 = []
        WaitingTimes2 = []
        ServiceTimes2 = []
        Qlens2 = []
        Utils2 = []

        for j in range(0, 20):
            Customers = []
            Mean_Time, Mean_Wait, Mean_Service_Time, Waits, Total_Times, Service_Times = foo(lambd, mu, simulation_time)
            mm1 = MM1()
            simulation_total_time, avg_queue_length, avg_utilization, nq = mm1.mm1(lambd, mu, 1000)
            ResponseTimes.append(Mean_Time)
            WaitingTimes.append(Mean_Wait)
            ServiceTimes.append(Mean_Service_Time)
            Qlens.append(avg_queue_length)
            Utils.append(avg_utilization)

            s = mmc()
            s.mmc(lambd, mu, simulation_time)
            avg_inter_arrival_time = s.clock / s.num_arrivals
            avg_service_time_server1 = s.dep_sum1 / s.num_of_departures1
            avg_service_time_server2 = s.dep_sum2 / s.num_of_departures2
            utilization_server1 = s.dep_sum1 / s.clock
            utilization_server2 = s.dep_sum2 / s.clock
            avg_numbers_in_q = s.number_in_queue
            avg_waiting_time = s.total_wait_time
            total_service_time = avg_service_time_server1 + avg_service_time_server2
            response_time = avg_waiting_time + avg_service_time_server1
            total_utilization = utilization_server1 + utilization_server2

            ResponseTimes2.append(response_time)
            WaitingTimes2.append(avg_waiting_time)
            ServiceTimes2.append(avg_service_time_server1)
            Qlens2.append(avg_numbers_in_q)
            Utils2.append(utilization_server1)

            a = pd.Series([
                lambd,
                avg_inter_arrival_time,
                avg_service_time_server1,
                avg_service_time_server2,
                utilization_server1,
                utilization_server2,
                avg_numbers_in_q,
                avg_waiting_time,
                response_time,
                total_utilization],
                index=df.columns)
            df = df.append(a, ignore_index=True)

        print("Confidence interval for MMC")
        print("Confidence level : 99%")
        print("Confidence interval for W")
        lw2, uw2 = calculate_Confidence_interval(WaitingTimes2)
        print("lower bound : " + str(lw2) + " upper bound : " + str(uw2))
        print("Confidence interval for R")
        lr2, ur2 = calculate_Confidence_interval(ResponseTimes2)
        print("lower bound : " + str(lr2) + " upper bound : " + str(ur2))
        print("Confidence interval for S")
        ls2, us2 = calculate_Confidence_interval(ServiceTimes2)
        print("lower bound : " + str(ls2) + " upper bound : " + str(us2))
        print("Confidence interval for Nw")
        lq2, uq2 = calculate_Confidence_interval(Qlens2)
        print("lower bound : " + str(lq2) + "  upper bound : " + str(uq2))
        print("Confidence interval for U")
        lu2, uu2 = calculate_Confidence_interval(Utils2)
        print("lower bound : " + str(lu2) + "  upper bound : " + str(uu2))

        Customers = []
        Mean_Time, Mean_Wait, Mean_Service_Time, Waits, Total_Times, Service_Times = foo(lambd, mu, simulation_time)
        mm1 = MM1()
        simulation_total_time, avg_queue_length, avg_utilization, nq = mm1.mm1(lambd, mu, 1000)

        print("Confidence interval for MM1")
        print("Confidence level : 99%")
        print("Confidence interval for W")
        lw, uw = calculate_Confidence_interval(Waits)
        print("lower bound : " + str(lw) + " upper bound : " + str(uw))
        print("Confidence interval for R")
        lr, ur = calculate_Confidence_interval(Total_Times)
        print("lower bound : " + str(lr) + " upper bound : " + str(ur))
        print("Confidence interval for S")
        ls, us = calculate_Confidence_interval(Service_Times)
        print("lower bound : " + str(ls) + " upper bound : " + str(us))
        print("Confidence interval for Nw")
        lq, uq = calculate_Confidence_interval(nq)
        print("lower bound : " + str(lq) + "  upper bound : " + str(uq))
        print("Confidence interval for U")
        lu, uu = calculate_Confidence_interval(Utils)
        print("lower bound : " + str(lu) + "  upper bound : " + str(uu))

        s = mmc()
        s.mmc(lambd, mu, simulation_time)
        avg_inter_arrival_time = s.clock / s.num_arrivals
        avg_service_time_server1 = s.dep_sum1 / s.num_of_departures1
        avg_service_time_server2 = s.dep_sum2 / s.num_of_departures2
        utilization_server1 = s.dep_sum1 / s.clock
        utilization_server2 = s.dep_sum2 / s.clock
        avg_numbers_in_q = s.number_in_queue
        avg_waiting_time = s.total_wait_time
        total_service_time = avg_service_time_server1 + avg_service_time_server2
        response_time = avg_waiting_time + avg_service_time_server1
        total_utilization = utilization_server1 + utilization_server2
        print("num_of_departures1 : " + str(s.num_of_departures1))
        print("num_of_departures2 : " + str(s.num_of_departures2))
        print("num_in_system : " + str(s.num_in_system))
        print("number_in_queue : " + str(s.number_in_queue))

        a3 = pd.Series([
            round(lambd, 4),
            round(mu, 4),
            s.num_in_system,
            s.num_of_departures1,
            s.num_of_departures2,
            round(rho2, 4),
            round(utilization_server1, 4),
            round(lu2, 4),
            round(uu2, 4),
            round(S2, 4),
            round(avg_service_time_server1, 4),
            round(ls2, 4),
            round(us2, 4),
            round(W2, 4),
            round(avg_waiting_time, 2),
            round(lw2, 4),
            round(uw2, 4),
            round(T2, 4),
            round(response_time, 4),
            round(lr2, 4),
            round(ur2, 4),
            round(N_w2, 4),
            round(avg_numbers_in_q, 4),
            round(lq2, 4),
            round(uq2, 4),
        ],
            index=df3.columns)
        df3 = df3.append(a3, ignore_index=True)

        a2 = pd.Series([
            round(lambd, 4),
            round(mu, 4),
            round(rho, 4),
            round(avg_utilization, 4),
            round(lu, 4),
            round(uu, 4),
            round(S, 4),
            round(Mean_Service_Time, 4),
            round(ls, 4),
            round(us, 4),
            round(W, 4),
            round(Mean_Wait, 4),
            round(lw, 4),
            round(uw, 4),
            round(T, 4),
            round(Mean_Time, 4),
            round(lr, 4),
            round(ur, 4),
            round(N_w, 4),
            round(avg_queue_length, 4),
            round(lq, 4),
            round(uq, 4),
            round(simulation_total_time, 4)
        ],
            index=df2.columns)
        df2 = df2.append(a2, ignore_index=True)

        i = i + 1
        # output to file
        outrow1 = []
        outrow1.append(i)
        outrow1.append(lambd)
        outrow1.append(Mean_Time)
        outrow1.append(Theoretical_result)
        lambd += 0.1  # increment lambda to next arrival rate
        output1.writerow(outrow1)

        rho_stored.append(rho)
        rho2_stored.append(rho2)

        # mm1
        RR_stored.append(T)
        R_stored.append(Mean_Time)
        LR_stored.append(lr)
        UR_stored.append(ur)

        WW_stored.append(W)
        W_stored.append(Mean_Wait)
        LW_stored.append(lw)
        UW_stored.append(uw)

        SS_stored.append(S)
        S_stored.append(Mean_Service_Time)
        LS_stored.append(ls)
        US_stored.append(us)

        N_stored.append(len(Customers))
        NN_stored.append(N)

        U_stored.append(avg_utilization)
        UU_stored.append(rho)
        LowU_stored.append(lu)
        upU_stored.append(uu)

        Nw_stored.append(avg_queue_length)
        Nww_stored.append(N_w)
        LNw_stored.append(lq)
        UNw_stored.append(uq)

        # mmc
        RR2_stored.append(T2)
        R2_stored.append(response_time)
        LR2_stored.append(lr2)
        UR2_stored.append(ur2)

        WW2_stored.append(W2)
        W2_stored.append(avg_waiting_time)
        LW2_stored.append(lw2)
        UW2_stored.append(uw2)

        SS2_stored.append(S2)
        S2_stored.append(avg_service_time_server1)
        LS2_stored.append(ls2)
        US2_stored.append(us2)

        U2_stored.append(utilization_server1)
        UU2_stored.append(rho2)
        LowU2_stored.append(lu2)
        upU2_stored.append(uu2)

        Nw2_stored.append(avg_numbers_in_q)
        Nww2_stored.append(N_w2)
        LNw2_stored.append(lq2)
        UNw2_stored.append(uq2)

    df.to_excel('main4/results.xlsx')
    df2.to_excel('main4/MM1_Performance_Parameters.xlsx')
    df3.to_excel('main4/MMC_Performance_Parameters.xlsx')

    outfile1.close()  # file close

    # mm1
    plt.plot(rho_stored, N_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, NN_stored, color='yellow', label="analytical results curve")
    plt.xlabel("system load rho")
    plt.ylabel("Avg numbers in system")
    plt.title("Avg Number of Customers in System curve MM1 Figure 1")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, R_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, RR_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LR_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, UR_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('Utilization - Rho - System load')
    plt.ylabel('Response time')
    plt.title("Avg Response time of System curve MM1 Figure 2")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, W_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, WW_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LW_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, UW_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('Utilization - Rho - System load')
    plt.ylabel('Waiting time')
    plt.title("Avg Waiting time of System curve MM1 Figure 3")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, S_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, SS_stored, color='yellow', label="analytic results curve")
    plt.plot(rho_stored, LS_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, US_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('Utilization - Rho - System load')
    plt.ylabel('Service time')
    plt.title("Avg Service time of System curve MM Figure 4")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, U_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, UU_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LowU_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, upU_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('System load rho')
    plt.ylabel('utilization')
    plt.title('Utilization Curve MM1 Figure 5')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, Nw_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, Nww_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LNw_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, UNw_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('System load rho')
    plt.ylabel('Avg nums in queue')
    plt.title('Avg number of customers in waiting line Curve MM1 Figure 6')
    plt.grid(True)
    plt.legend()
    plt.show()

    # mmc

    plt.plot(rho_stored, R2_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, RR2_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LR2_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, UR2_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('Utilization - Rho - System load')
    plt.ylabel('Response time')
    plt.title("Avg Response time of System curve MMC Figure 1")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, W2_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, WW2_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LW2_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, UW2_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('Utilization - Rho - System load')
    plt.ylabel('Waiting time')
    plt.title("Avg Waiting time of System curve MMC Figure 2")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, S2_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, SS2_stored, color='yellow', label="analytic results curve")
    plt.plot(rho_stored, LS2_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, US2_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('Utilization - Rho - System load')
    plt.ylabel('Service time')
    plt.title("Avg Service time of System curve MMC Figure 3")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, U2_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, UU2_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LowU2_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, upU2_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('System load rho')
    plt.ylabel('utilization')
    plt.title('Utilization Curve MMC Figure 4')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(rho_stored, Nw2_stored, color='blue', label="simulation results curve")
    plt.plot(rho_stored, Nww2_stored, color='yellow', label="analytical results curve")
    plt.plot(rho_stored, LNw2_stored, color='red', linestyle='dashed', label="lower bound curve")
    plt.plot(rho_stored, UNw2_stored, color='green', linestyle='dashed', label="upper bound curve")
    plt.xlabel('System load rho')
    plt.ylabel('Avg nums in queue')
    plt.title('Avg number of customers in waiting line Curve MMC Figure 5')
    plt.grid(True)
    plt.legend()
    plt.show()
