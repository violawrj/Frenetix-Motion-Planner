__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import os
import time
from multiprocessing import Process, Queue, Event
from queue import Empty
from typing import Optional, List

import numpy as np
from commonroad.scenario.scenario import Scenario

from cr_scenario_handler.utils.agent_status import AgentStatus, TIMEOUT


def determine_collision_type(agent, scenario, timestep):
    """
    Classify the ego's collision relative to the nearest dynamic obstacle.

    Returns (collision_type, colliding_object_id) where collision_type is one of
    "front_collision", "rear_collision", "side_collision", "boundary_collision", "unknown".
    """
    try:
        ego_state = agent.record_state_list[-1] if agent.record_state_list else None
        if ego_state is None:
            return ("unknown", None)

        ego_pos = np.array([ego_state.position[0], ego_state.position[1]])
        ego_orientation = ego_state.orientation

        ego_length = 4.508
        ego_width = 1.610

        closest_distance = float('inf')
        closest_obs_id = None
        closest_collision_type = None

        for obs in scenario.dynamic_obstacles:
            if obs.obstacle_id == agent.id:
                continue

            try:
                obs_state = obs.state_at_time(timestep)
                if obs_state is None:
                    continue

                obs_pos = np.array([obs_state.position[0], obs_state.position[1]])
                relative_pos = obs_pos - ego_pos
                distance = np.linalg.norm(relative_pos)

                if distance > 10.0:
                    continue

                cos_theta = np.cos(-ego_orientation)
                sin_theta = np.sin(-ego_orientation)
                rotation_matrix = np.array([[cos_theta, -sin_theta],
                                            [sin_theta, cos_theta]])
                local_relative_pos = rotation_matrix @ relative_pos

                d_long = local_relative_pos[0]
                d_lat = local_relative_pos[1]

                front_threshold = ego_length / 2
                rear_threshold = -ego_length / 2
                side_threshold = ego_width / 2

                if d_long > front_threshold:
                    collision_type = "front_collision"
                elif d_long < rear_threshold:
                    collision_type = "rear_collision"
                elif abs(d_lat) > side_threshold:
                    collision_type = "side_collision"
                else:
                    collision_type = "front_collision"

                if distance < closest_distance:
                    closest_distance = distance
                    closest_obs_id = obs.obstacle_id
                    closest_collision_type = collision_type

            except Exception:
                continue

        if closest_obs_id is not None:
            return (closest_collision_type, closest_obs_id)

        return ("boundary_collision", None)

    except Exception:
        return ("unknown", None)


class AgentBatch(Process):

    def __init__(self, agents, global_timestep: int, msg_logger, log_path: str, mod_path: str,
                 in_queue: Optional[Queue] = None, out_queue: Optional[Queue] = None, event: Optional[Event] = None):
        """Batch of agents.

        Manages the Agents in this batch, and communicates dummy obstacles,
        predictions, and plotting data with the main simulation.

        If multiprocessing is enabled, all batches are processed in parallel,
        execution inside one batch is sequential.

        :param agents: list of agent objects in this batch.
        :param global_timestep: initial simulation timestep
        :param msg_logger: logger
        :param log_path: Base path of the log files.
        :param mod_path: Path of the working directory of the planners.
        :param in_queue: Queue the batch receives data from (None for serial execution).
        :param out_queue: Queue the batch sends data to (None for serial execution).
        """

        super().__init__()

        self.msg_logger = msg_logger

        # Initialize queues
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.event = event

        self.log_path = log_path
        self.mod_path = mod_path

        # Initialize batch
        self.global_timestep = global_timestep
        self.agent_list = agents
        self.agent_ids = [agent.id for agent in agents]

        # list of all active agents
        self.running_agent_list = []
        # list of all finished agents
        self.terminated_agent_list = []

        self.process_times = dict()

        self.finished = False

        # initialize communication dict
        self.out_queue_dict = dict.fromkeys(self.agent_ids, {})

        self.latest_starting_time = max([agent.current_timestep for agent in self.agent_list]) if self.agent_list else 0

        # dict of all agents with corresponding starting times
        self.agent_dict = {timestep: [agent for agent in self.agent_list if agent.current_timestep == timestep] for
                           timestep in range(self.latest_starting_time+1)}

    def run(self):
        """ Main function of the agent batch when running in a separate process.

        Receives necessary information from the main simulation, and performs one simulation step in the batch,
        sends back agent updates and the batch status


        The messages exchanged with the main simulation are:
        in_queue (args):
            -input for self.step_simulation: [scenario, global_timestep, global_predictions, colliding_agents]
            - if all agents are terminated: any synchronization message triggering the termination of this batch.
        out_queue:
            - out_queue_dict: current agent update (see self._update_batch())
            - batch status

        """
        while True:
            # Receive the next predictions
            # Synchronize agents
            # receive dummy obstacles and outdated agent list
            start_time = time.perf_counter()

            try:
                args = self.in_queue.get(block=True, timeout=TIMEOUT)
            except Empty:
                self.event.set()
                self.msg_logger.error(f"Batch {self.name}: Timeout waiting for "
                                      f"{'simulation' if self.finished else 'agent'} updates!")

                raise RuntimeError(f"Batch {self.name}: Timeout waiting for "
                                   f"{'simulation' if self.finished else 'agent'} updates!")

            if self.event.is_set():
                self.msg_logger.error(f"Batch {self.name}: Termination event is triggered!")
                break

            sync_time_in = time.perf_counter() - start_time

            if self.finished:
                # if batch finished, postprocess agents (currently only make_gif())
                for agent in self.terminated_agent_list:
                    agent.make_gif()
                self.msg_logger.critical(f"Batch {self.name}: Simulation of the batch finished!")
                break

            else:
                # simulate next step
                self.step_simulation(*args)

                syn_time_out = time.perf_counter()
                # send agent updates to simulation
                self.out_queue.put(self.out_queue_dict)

            self.process_times.update({"sync_time_out": time.perf_counter() - syn_time_out,
                                       "process_iteration_time": time.perf_counter() - start_time,
                                       "sync_time_in": sync_time_in})

            self.out_queue.put([self.finished, self.process_times])


            if self.event.is_set():
                self.msg_logger.error(f"Batch {self.name}: Termination event is triggered!")
                break
            #print(self.name, self.event.is_set())

    def step_simulation(self, scenario, global_timestep, global_predictions, colliding_agents):
        """Simulate the next timestep.

        Adds later starting agents to running list,
        updates agents with current scenario, prediction and colliding agent-IDs and
        calls the step function of the agents.
        After each step, the status of the agents within the batch is updated and the batch checks for its completion.

        :param scenario: current valid (global) scenario representation
        :param global_timestep: current global timestep
        :param global_predictions: prediction dict with all obstacles within the scenario
        :param colliding_agents: list with IDs of agents that collided in the prev. timestep
        """

        step_time = time.perf_counter()
        self.msg_logger.debug(f"Stepping Batch {self.name}")
        # update batch timestep
        self.global_timestep = global_timestep
        # add agents if they enter the scenario
        if self.global_timestep <= self.latest_starting_time:
            self.running_agent_list.extend(self.agent_dict[self.global_timestep])

        # update agents
        self._update_agents(scenario, global_predictions, colliding_agents)

        # step simulation
        single_step_time = time.perf_counter()
        self._step_agents(global_timestep)
        single_step_time = time.perf_counter() - single_step_time

        # update batch
        self._update_batch()
        self.msg_logger.debug(f"Batch {self.name} updated")
        # check for batch completion
        self._check_completion()
        self.msg_logger.debug(f"Batch {self.name} completion checked")
        self.process_times.update({"sim_step_time": time.perf_counter() - step_time,
                                   "agent_planning_time": single_step_time,
                                   })

    def _update_agents(self, scenario: Scenario, global_predictions: dict,  colliding_agents: List):
        for agent in self.running_agent_list:
            # update agent if he collided and update predictions and scenario
            collision = True if agent.id in colliding_agents else False
            agent.update_agent(scenario, self.global_timestep, global_predictions, collision)

    def _step_agents(self, global_timestep):
        for agent in self.running_agent_list:
            # plan one step in each agent
            agent.step_agent(global_timestep)

    def _update_batch(self):
        """
        update agent lists and prepare dict to send to simulation
        Current agent update:
            - Agent status
            - Agent collision objects for the global collision check
            - Agent vehicle history for visualization
        """
        for agent in reversed(self.running_agent_list):
            if agent.status > AgentStatus.RUNNING:
                self.terminated_agent_list.append(agent)
                self.running_agent_list.remove(agent)
                with (open(os.path.join(self.mod_path, "logs", "score_overview.csv"), 'a') as file):
                    msg = "Success" if agent.status == AgentStatus.COMPLETED_SUCCESS else "Failed"

                    collision_type = ""
                    colliding_object_id = ""
                    if agent.status == AgentStatus.COLLISION:
                        collision_type, colliding_object_id = determine_collision_type(
                            agent, agent.scenario, agent.current_timestep
                        )
                        colliding_object_id = str(colliding_object_id) if colliding_object_id is not None else ""

                    line = str(agent.scenario.scenario_id) + ";" + str(agent.id) + ";" + str(agent.current_timestep) + ";" + \
                           str(agent.status) + ";" + str(agent.agent_state.message) + ";" + msg + ";" + \
                           str(collision_type) + ";" + str(colliding_object_id) + "\n"
                    file.write(line)

            self.out_queue_dict[agent.id] = {"agent_state": agent.agent_state,
                                             "collision_objects": agent.collision_objects[-1],
                                             "vehicle_history": agent.vehicle_history[-1],
                                             "record_state_list": agent.record_state_list[-1],
                                             "record_input_list": agent.record_input_list[-1],
                                             "planning_times": agent.planning_times[-1],
                                             }

    def _check_completion(self):
        """
        check for completion of all agents in this batch.
        """
        self.finished = all([i.status > AgentStatus.RUNNING for i in self.agent_list])

