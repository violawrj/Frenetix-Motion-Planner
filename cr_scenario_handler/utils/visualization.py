__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import os
from typing import List, Union, Dict
from PIL import Image

import matplotlib
from matplotlib import pyplot as plt
import imageio.v3 as iio
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import State, CustomState
from commonroad.planning.planning_problem import PlanningProblem
from cr_scenario_handler.utils.agent_status import AgentStatus
# commonroad_dc
from commonroad_dc import pycrcc

from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams, ShapeParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle

from cr_scenario_handler.utils.configuration import Configuration

from wale_net_lite.visualization import draw_uncertain_predictions
from pathlib import Path

import json

# Color palette for plotting
colors_spec = ["#e3af22", "#d6e322", "#96e322", "#55e322", "#22e32f",
               "#22e36f", "#22e3af", "#22d6e3", "#2296e3", "#2255e3",
               "#2f22e3", "#6f22e3", "#af22e3", "#e322d6", "#e32296"]
darkcolors = ["#9c0d00", "#8f9c00", "#5b9c00", "#279c00", "#009c0d",
              "#009c41", "#009c75", "#008f9c", "#005b9c", "#00279c",
              "#0d009c", "#41009c", "#75009c", "#9c008f", "#9c005b"]
lightcolors = ["#ffd569", "#f8ff69", "#c6ff69", "#94ff69", "#69ff70",
               "#69ffa3", "#69ffd5", "#69f8ff", "#69c6ff", "#6993ff",
               "#7069ff", "#a369ff", "#d569ff", "#ff69f8", "#ff69c5"]


def visualize_agent_at_timestep(scenario: Scenario, planning_problem: PlanningProblem, ego: DynamicObstacle,
                                timestep: int, config, log_path: str, traj_set=None, optimal_traj=None,
                                ref_path: np.ndarray = None, rnd: MPRenderer = None, predictions: dict = None,
                                plot_window: int = None, visible_area=None, occlusion_map=None,
                                behavior_module_state: {} = None, save: bool = False, show: bool = False,
                                gif: bool = False, replanning_counter: int = 0):
    """
    Function to visualize planning result from the reactive planner for a given time step
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param ego: Ego vehicle as CommonRoad DynamicObstacle object
    :param pos: positions of planned trajectory [(nx2) np.ndarray]
    :param timestep: current time step of scenario to plot
    :param log_path: Log path where to save the plots
    :param config_visu: Configuration object for plot/save settings
    :param traj_set: List of sampled trajectories (optional)
    :param optimal_traj: Optimal Trajectory selected
    :param ref_path: Reference path for planner as polyline [(nx2) np.ndarray] (optional)
    :param rnd: MPRenderer object (optional: if none is passed, the function creates a new renderer object; otherwise it
    :param predictions: Predictions used to run the planner
    :param plot_window: Window size to plot (optional)
    :param visible_area: Visible Area for plotting (optional)
    :param occlusion_map: Occlusion map information to plot (optional)
    will visualize on the existing object)
    :param behavior_module_state: state of the behavior module as dict (optional)
    :param save_path: Path to save plot as .png (optional)
    """
    # Only create renderer if not passed in
    if rnd is None:
        # Assuming ego.prediction.trajectory.state_list[0].position returns a constant value
        ego_start_pos = ego.prediction.trajectory.state_list[0].position
        if plot_window > 0:
            plot_limits = [-plot_window + ego_start_pos[0], plot_window + ego_start_pos[0],
                           -plot_window + ego_start_pos[1], plot_window + ego_start_pos[1]]
            rnd = MPRenderer(plot_limits=plot_limits, figsize=(10, 10))
        else:
            rnd = MPRenderer(figsize=(20, 10))

    # set ego vehicle draw params
    ego_params = DynamicObstacleParams()
    ego_params.time_begin = timestep
    ego_params.draw_icon = config.visualization.draw_icons
    ego_params.show_label = True
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#2255e3"  # Change facecolor to blue
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#003359"  # Change edgecolor to dark blue
    ego_params.vehicle_shape.occupancy.shape.zorder = 50
    ego_params.vehicle_shape.occupancy.shape.opacity = 1

    # Leave this in, because our code draws limited custom trajectories for each timestep
    ego_params.trajectory.draw_trajectory = False

    obs_params = MPDrawParams()
    # obs_params.traffic_light.draw_traffic_lights = False
    # obs_params.lanelet_network.traffic_light.draw_traffic_lights = False
    # obs_params.traffic_sign.draw_traffic_signs = False
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = config.visualization.draw_icons
    obs_params.dynamic_obstacle.show_label = True
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
    # obs_params.dynamic_obstacle.trajectory.draw_trajectory = False

    obs_params.static_obstacle.show_label = False
    obs_params.static_obstacle.occupancy.shape.facecolor = "#a30000"
    obs_params.static_obstacle.occupancy.shape.edgecolor = "#756f61"

    # visualize scenario, planning problem, ego vehicle
    scenario.draw(rnd, draw_params=obs_params)
    planning_problem.draw(rnd)
    ego.draw(rnd, draw_params=ego_params)

    rnd.render()

    # visualize optimal trajectory
    optimal_traj_handle = None
    if optimal_traj:
        optimal_traj_positions = np.array([(state.position[0], state.position[1]) for state in optimal_traj.state_list[0+replanning_counter:]])
        optimal_traj_handle, = rnd.ax.plot(optimal_traj_positions[:, 0], optimal_traj_positions[:, 1], 'kx-', markersize=1.5, zorder=21, linewidth=2.0, label='Optimal Trajectory')

    if traj_set is not None and replanning_counter == 0:
        valid_traj = [obj for obj in traj_set if obj.valid is True and obj.feasible is True]
        invalid_traj = [obj for obj in traj_set if obj.valid is False or obj.feasible is False]

        valid_step = max(1, int(len(valid_traj)))  # Ensure valid_step is at least 1

        # Lists to store plotted lines and their labels for the legend
        legend_handles = []
        legend_labels = []
        all_trajectories_data = {} # List to store data for JSON export
        all_cost_data = {} # List to store cost data for JSON export

        # Process valid trajectories
        for idx in range(0, len(valid_traj), valid_step):
            val = valid_traj[idx]
            
            # --- JSON Data Extraction for valid trajectories ---
            # Convert numpy arrays to lists for JSON serialization if necessary
            x_coords = val.cartesian.x.tolist() if hasattr(val.cartesian.x, 'tolist') else val.cartesian.x
            y_coords = val.cartesian.y.tolist() if hasattr(val.cartesian.y, 'tolist') else val.cartesian.y
            velocity = val.cartesian.v.tolist() if hasattr(val.cartesian.v, 'tolist') else val.cartesian.v
            acceleration = val.cartesian.a.tolist() if hasattr(val.cartesian.a, 'tolist') else val.cartesian.a
            theta = val.cartesian.theta.tolist() if hasattr(val.cartesian.theta, 'tolist') else val.cartesian.theta
            trajectory_data = {
                'x': [round(coord, 2) for coord in x_coords[::5]],
                'y': [round(coord, 2) for coord in y_coords[::5]],
                'velocity': [round(v, 2) for v in velocity[::5]],
                'acceleration': [round(a, 2) for a in acceleration[::5]],
                'heading_radian': [round(t, 2) for t in theta[::5]],
            }
            all_trajectories_data[f'trajectory_id:{val.uniqueId}'] = trajectory_data

            # Format cost data to two decimal places
            cost_data = {
                'cost': round(val.cost, 2), # Assuming val.cost is a single float
                'costMap': {} # Initialize an empty dictionary for formatted costMap
            }

            
            for key, value_list in val.costMap.items():
                if isinstance(value_list, (list, tuple)):
                    
                    formatted_list = [round(item, 2) for item in value_list if isinstance(item, (int, float))]
                    cost_data['costMap'][key] = formatted_list
                elif isinstance(value_list, (int, float)):
                    
                    cost_data['costMap'][key] = round(value_list, 2)
                else:
                    cost_data['costMap'][key] = value_list # Or str(value_list) if you prefer string conversion
            all_cost_data[f'trajectory_id:{val.uniqueId}'] = cost_data
            # --- End of JSON Data Extraction ---

            line, = plt.plot(val.cartesian.x, val.cartesian.y,
                            zorder=60, linewidth=1.2, alpha=1, linestyle='--', picker=False)
            legend_handles.append(line)
            legend_labels.append(f"{val.uniqueId}")

        # Process invalid trajectories
        step = int(len(invalid_traj) / 2) if len(invalid_traj) > 0 else 1
        for ival in range(0, len(invalid_traj), step):
            in_val = invalid_traj[ival]

            # --- JSON Data Extraction for invalid trajectories ---
            x_coords = in_val.cartesian.x.tolist() if hasattr(in_val.cartesian.x, 'tolist') else in_val.cartesian.x
            y_coords = in_val.cartesian.y.tolist() if hasattr(in_val.cartesian.y, 'tolist') else in_val.cartesian.y
            velocity = in_val.cartesian.v.tolist() if hasattr(in_val.cartesian.v, 'tolist') else in_val.cartesian.v
            acceleration = in_val.cartesian.a.tolist() if hasattr(in_val.cartesian.a, 'tolist') else in_val.cartesian.a
            theta = in_val.cartesian.theta.tolist() if hasattr(in_val.cartesian.theta, 'tolist') else in_val.cartesian.theta

            trajectory_data = {
                'x': [round(coord, 2) for coord in x_coords[::5]],
                'y': [round(coord, 2) for coord in y_coords[::5]],
                'velocity': [round(v, 2) for v in velocity[::5]],
                'acceleration': [round(a, 2) for a in acceleration[::5]],
                'heading_radian': [round(t, 2) for t in theta[::5]],
            }
            all_trajectories_data[f'trajectory_id:{in_val.uniqueId}'] = trajectory_data

            
            cost_data = {
                'cost': round(in_val.cost, 2), # Assuming val.cost is a single float
                'costMap': {} # Initialize an empty dictionary for formatted costMap
            }

            
            for key, value_list in in_val.costMap.items():
                if isinstance(value_list, (list, tuple)):
                    
                    formatted_list = [round(item, 2) for item in value_list if isinstance(item, (int, float))]
                    cost_data['costMap'][key] = formatted_list
                elif isinstance(value_list, (int, float)):
                    
                    cost_data['costMap'][key] = round(value_list, 2)
                else:
                    
                    cost_data['costMap'][key] = value_list # Or str(value_list) if you prefer string conversion
            all_cost_data[f'trajectory_id:{in_val.uniqueId}'] = cost_data
            # --- End of JSON Data Extraction ---
            
            line, = plt.plot(in_val.cartesian.x, in_val.cartesian.y,
                            zorder=60, linewidth=0.8, alpha=1, linestyle='--', picker=False)
            legend_handles.append(line)
            legend_labels.append(f"{in_val.uniqueId}")

        # Add optimal trajectory to legend if it exists
        # if optimal_traj_handle is not None:
        #     legend_handles.insert(0, optimal_traj_handle)
        #     legend_labels.insert(0, "Optimal Trajectory")
        # 
        # legend = plt.legend(legend_handles, legend_labels, title="Trajectories", loc='best', fontsize='small')
        # legend.set_zorder(100)
        merged_data = {}
        for trajectory_id, trajectory_info in all_trajectories_data.items():
           
            merged_data[trajectory_id] = {
                'trajectory': trajectory_info,
                'cost': {},
            }
            
            
            if trajectory_id in all_cost_data:
                merged_data[trajectory_id]['cost'] = all_cost_data[trajectory_id]


        # --- Save all collected data to a JSON file ---
        def save_json_robustly(data, output_filepath: str):
            """
            Saves a dictionary to a JSON file, creating parent directories if they don't exist.

            Args:
                data (dict): The dictionary to save.
                output_filepath (str): The full path to the output JSON file.
            """
            # Use the pathlib module to handle file paths in an object-oriented way
            path = Path(output_filepath)
            
            # Create the parent directories of the file.
            # The 'parents=True' argument ensures all parent directories are created.
            # The 'exist_ok=True' argument prevents an error if the directories already exist.
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the data to the JSON file
            with path.open('w') as f:
                json.dump(data, f, indent=4)
            print(f"Successfully saved data to: {output_filepath}")
        # Use the robust function to save each file
        # Modified to store in different location compared to before
        output_filename = f'{log_path}/trajectory_cost/trajectory_cost_{timestep}.json'
        output_filename_trajectory = f'{log_path}/trajectory/trajectory_{timestep}.json'
        output_filename_cost = f'{log_path}/cost/cost_{timestep}.json'

        save_json_robustly(merged_data, output_filename)
        save_json_robustly(all_trajectories_data, output_filename_trajectory)
        save_json_robustly(all_cost_data, output_filename_cost)

        print(f"Trajectory data successfully saved to {output_filename_trajectory}")
        print(f"Cost data successfully saved to {output_filename_cost}")
    # elif optimal_traj_handle is not None:
    #     # If no sampled trajectories but optimal trajectory exists, show legend for it
    #     plt.legend(handles=[optimal_traj_handle], labels=["Optimal Trajectory"], loc='best', fontsize='small')

    # visualize predictions
    # if predictions is not None:
    #     predictions_copy = predictions
    #     if ego.obstacle_id in predictions.keys():
    #         predictions_copy = predictions.copy()
    #         # Remove the entry with key 60000 from the copy
    #         predictions_copy.pop(ego.obstacle_id, None)
    #     if config.prediction.mode == "walenet":
    #         draw_uncertain_predictions(predictions_copy, rnd.ax)

    # # visualize reference path
    # if ref_path is not None:
    #     rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
    #                 label='reference path')

    # visualize behavior states
    if behavior_module_state is not None:
        if config.behavior.use_behavior_planner and config.behavior.visualize_states:
            rnd.f.texts = []  # remove the texts, otherwise they are stacking
            behavior_text_left = ""
            if config.behavior.visualization_mode == "SLIM":
                behavior_text_left = (
                    f"behavior_state_static: {str(behavior_module_state.get('behavior_state_static'))}\n"
                    f"behavior_state_dynamic: {str(behavior_module_state.get('behavior_state_dynamic'))}\n"
                )
            if config.behavior.visualization_mode in ["BASIC", "EXTENDED", "FULL"]:
                behavior_text_left = (
                    f"street_setting: {str(behavior_module_state.get('street_setting'))}\n"
                    f"behavior_state_static: {str(behavior_module_state.get('behavior_state_static'))}\n"
                    f"situation_state_static: {str(behavior_module_state.get('situation_state_static'))}\n"
                    f"behavior_state_dynamic: {str(behavior_module_state.get('behavior_state_dynamic'))}\n"
                    f"situation_state_dynamic: {str(behavior_module_state.get('situation_state_dynamic'))}\n"
                )
            if config.behavior.visualization_mode in ["EXTENDED", "FULL"]:
                behavior_text_left += (
                    f"goal_velocity: {behavior_module_state.get('goal_velocity'):.2f}\n"
                    f"desired_velocity: {behavior_module_state.get('desired_velocity'):.2f}\n"
                    f"velocity: {behavior_module_state.get('velocity'):.2f}\n"
                )
            if config.behavior.visualization_mode in ["FULL"]:
                if behavior_module_state.get('TTC') is None:
                    ttc_str = f"None"
                else:
                    ttc_str = f"{behavior_module_state.get('TTC'):.2f}"
                behavior_text_left += (
                    f"TTC: {ttc_str}\n"
                    f"MAX: {behavior_module_state.get('MAX'):.2f}\n"
                    f"stop_point_dist: {behavior_module_state.get('stop_point_dist'):.2f}\n"
                    f"desired_velocity_stop_point: {behavior_module_state.get('desired_velocity_stop_point'):.2f}\n"
                    f"stop_point_mode: {str(behavior_module_state.get('stop_point_mode'))}\n"
                    f"lane_change_target_lanelet_id: {str(behavior_module_state.get('lane_change_target_lanelet_id'))}\n"
                    f"slowing_car_for_traffic_light: {str(behavior_module_state.get('slowing_car_for_traffic_light'))}\n"
                    f"waiting_for_green_light: {str(behavior_module_state.get('waiting_for_green_light'))}\n"
                    f"condition_factor: {behavior_module_state.get('condition_factor'):.2f}\n"
                    f"lon_dyn_cond_factor: {behavior_module_state.get('lon_dyn_cond_factor'):.2f}\n"
                    f"lat_dyn_cond_factor: {behavior_module_state.get('lat_dyn_cond_factor'):.2f}\n"
                    f"visual_cond_factor: {behavior_module_state.get('visual_cond_factor'):.2f}\n"
                )

            rnd.f.text(0.15, 0.85, behavior_text_left, ha='left', va='top', fontsize=12)

    # save as .png file
    if config.visualization.save_plots or save or gif:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if config.visualization.save_gif or gif:
            plt.axis('off')
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300)
        else:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.svg", format='svg')

    # show plot
    if config.visualization.show_plots or show:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)


def visualize_multiagent_scenario_at_timestep(scenario: Scenario,
                                              agent_list: List,
                                              timestep: int,
                                              config: Configuration, log_path: str,
                                              orig_pp_id: List=None,
                                              predictions: dict = None,
                                              rnd: MPRenderer = None,
                                              plot_window: int = None,
                                              save: bool = False, show: bool = False, gif: bool = False):
    """
    Function to visualize planning result from the reactive planner for a given time step
    for all agents in a multiagent simulation.
    Replaces visualize_visualize_planner_at_timestep from visualization.py

    :param scenario: CommonRoad scenario object containing no dummy obstacles
    :param agent_list: Dummy obstacles for all agents. Assumed to include the recorded path in the trajectory
    :param timestep: Time step of the scenario to plot
    :param config: Configuration object for plot/save settings
    :param log_path: Path to save the plot to (optional, depending on the config)
    :param predictions: Dictionary of all predictions (optional)
    :param rnd: MPRenderer object (optional: if none is passed, the function creates a new renderer object;
                otherwise it will visualize on the existing object)
    :param plot_window: Size of the margin of the plot, or minimum distance between the
                        plot border and all agents (optional)
    """

    # create renderer object (if no existing renderer is passed)
    if rnd is None:
        rnd = MPRenderer(figsize=(20, 10))

        if plot_window is not None:
            # focus on window around all agents
            left = np.inf
            right = - np.inf
            top = - np.inf
            bottom = np.inf
            for agent in agent_list:
                try:
                    ini_pos = agent.planning_problem.initial_state.position
                    if hasattr(agent.planning_problem.goal.state_list[0].position, "shapes"):
                        goal_pos = agent.planning_problem.goal.state_list[0].position.shapes[0].center
                    elif hasattr(agent.planning_problem.goal.state_list[0].position, "center"):
                        goal_pos = agent.planning_problem.goal.state_list[0].position.center
                    # else:
                    #     raise ValueError

                    left = min(left, goal_pos[0], ini_pos[0])
                    right = max(right, goal_pos[0], ini_pos[0])
                    top = max(top, goal_pos[1], ini_pos[1])
                    bottom = min(bottom, goal_pos[1], ini_pos[1])
                except AttributeError:
                    pass

            rnd.plot_limits = [-plot_window + left,
                               plot_window + right,
                               -plot_window + bottom,
                               plot_window + top]

    # Set obstacle parameters
    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = config.visualization.draw_icons
    obs_params.dynamic_obstacle.show_label = config.visualization.show_labels
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

    obs_params.static_obstacle.show_label = config.visualization.show_labels
    obs_params.static_obstacle.occupancy.shape.facecolor = "#A30000"
    obs_params.static_obstacle.occupancy.shape.edgecolor = "#756F61"

    # visualize scenario
    scenario.draw(rnd, draw_params=obs_params)

    # Visualize agents and planning problems
    # for i in range(len(agent_list)):
    for agent in agent_list:
        if agent.status != AgentStatus.RUNNING:
            continue
        # set ego vehicle draw params
        ego_params = DynamicObstacleParams()
        ego_params.time_begin = timestep
        ego_params.draw_icon = config.visualization.draw_icons
        ego_params.show_label = config.visualization.show_labels

        # Use standard colors for single-agent plots
        if len(agent_list) == 1:
            ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
            ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
        elif orig_pp_id:
            if agent.id in orig_pp_id:
                ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
                ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
            else:
                obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
                obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
        else:
            ego_params.vehicle_shape.occupancy.shape.facecolor = \
                colors_spec[agent.id % len(colors_spec)]
            ego_params.vehicle_shape.occupancy.shape.edgecolor = \
                darkcolors[agent.id % len(darkcolors)]
        ego_params.vehicle_shape.occupancy.shape.zorder = 50
        ego_params.vehicle_shape.occupancy.shape.opacity = 1

        # Visualize planning problem and agent
        # if multiple_agents:
        #     planning_problems.find_planning_problem_by_id(agent.obstacle_id).draw(rnd)
        # else:

        agent.vehicle_history[-1].draw(rnd, draw_params=ego_params)
        agent.planning_problem.draw(rnd)
    rnd.render()
    for agent in agent_list:
        if agent.status != AgentStatus.RUNNING:
            continue
        # visualize optimal trajectory
        history = [i for i in agent.vehicle_history if i.initial_state.time_step == timestep]
        if history:
            history = history[0]

            if history.prediction is not None and config.visualization.draw_calc_traj:
                rnd.ax.plot([i.position[0] for i in history.prediction.trajectory.state_list],
                            [i.position[1] for i in history.prediction.trajectory.state_list],
                            color='k' , marker='x', markersize=1.5, zorder=21, linewidth=2.0, label='Optimal Trajectory')

        if agent.all_trajectories is not None and agent.config_planner.debug.draw_traj_set:
            valid_traj = [obj for obj in agent.all_trajectories if obj.valid is True and obj.feasible is True]
            invalid_traj = [obj for obj in agent.all_trajectories if obj.valid is False or obj.feasible is False]
            norm = matplotlib.colors.Normalize(
                vmin=0,
                vmax=len(valid_traj),
                clip=True,
            )
            mapper = cm.ScalarMappable(norm=norm, cmap=green_to_red_colormap())
            # step_inv = int(len(invalid_traj) / 50) if int(len(invalid_traj) / 50) > 2 else 1
            step_val = int(len(valid_traj)/50)  if int(len(valid_traj) / 50) > 2 else 1
            step_inv=step_val
            for idx in range(0, len(valid_traj), step_val):
                val = valid_traj[idx]
                if not val._coll_detected:
                    color = mapper.to_rgba(idx)
                    color = lightcolors[idx % len(lightcolors)]
                    plt.plot(val.cartesian.x, val.cartesian.y,
                             color=color, zorder=20, linewidth=0.8, linestyle='--', alpha=1.0, picker=False)
                else:
                    plt.plot(val.cartesian.x, val.cartesian.y,
                             color='cyan', zorder=20, linewidth=0.8, linestyle='--', alpha=0.8, picker=False)
            for ival in range(0, len(invalid_traj), step_inv):
                plt.plot(invalid_traj[ival].cartesian.x, invalid_traj[ival].cartesian.y,
                         color="#808080", zorder=19, linewidth=0.8, linestyle='--', alpha=0.4, picker=False)

        if agent.reference_path is not None and config.visualization.draw_reference_path:

            rnd.ax.plot(agent.reference_path[:, 0], agent.reference_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                        label='reference path')

    # visualize predictions
    if predictions is not None and config.visualization.draw_predictions:
        draw_uncertain_predictions(predictions, rnd.ax)

    # Add legend for labeled elements (optimal trajectory, reference path, etc.)
    # plt.legend(loc='best', fontsize='small')

    if save or gif:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if gif:
            plt.axis('off')
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300)
        else:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.svg", format='svg')

    if show:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)


def make_gif(scenario: Scenario, time_steps: Union[range, List[int]],
             log_path: str, duration: float = 0.1):
    """ Create an animated GIF from saved image files.

    Images are assumed to be saved as <log_path>/plots/<scenario_id>_<timestep>.png
    Does not check the plotting configuration in order to simplify independent
    configurations on agent and simulation level. This has to be done by the caller.

    :param scenario: CommonRoad scenario object.
    :param time_steps: List or range of time steps to include in the GIF
    :param log_path: Base path containing the plots folder with the input images
    :param duration: Duration of the individual frames (default: 0.1s)
    """

    print("Generating GIF")
    images = []
    filenames = []

    # directory, where single images are outputted (see visualize_planner_at_timestep())
    path_images = os.path.join(log_path, "plots")

    for step in time_steps:
        im_path = os.path.join(path_images, str(scenario.scenario_id) + f"_{step}.png")
        filenames.append(im_path)

    # First pass: find maximum dimensions across all images
    max_width = 0
    max_height = 0
    
    for filename in filenames:
        img = Image.open(filename)
        width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    
    # Second pass: pad all images to maximum dimensions (no distortion)
    for filename in filenames:
        img = Image.open(filename)
        width, height = img.size

        # If dimensions don't match, pad the image (no stretching!)
        if width != max_width or height != max_height:
            # Create a white canvas with max dimensions
            padded_img = Image.new('RGB', (max_width, max_height), 'white')
            # Paste the original image centered on the canvas
            offset = ((max_width - width) // 2, (max_height - height) // 2)
            padded_img.paste(img, offset)
            img = padded_img

        # Convert the PIL image to an array for imageio
        img_array = np.array(img)
        images.append(img_array)

    iio.imwrite(os.path.join(log_path, str(scenario.scenario_id) + ".gif"),
                images, duration=duration, loop=0)


def collision_vis(scenario: Scenario, ego_vehicle: DynamicObstacle, destination: str,
                  ego_harm: float, ego_type: ObstacleType, ego_v: float, ego_mass: float,
                  obs_harm: float, obs_type: ObstacleType, obs_v: float, obs_mass: float,
                  pdof: float, ego_angle: float, obs_angle: float,
                  time_step: int, modes: Dict, planning_problem: PlanningProblem = None,
                  global_path: np.ndarray = None, driven_traj: List[State] = None):
    """ Create a report for visualization of the collision and respective harm.

    Creates a collision report and saves the file in the destination folder
    in a subdirectory called "collisions".
    Replaces risk_assessment.visualization.collision_visualization.collision_vis
    to allow using the full ego obstacle.

    :param scenario: Considered Scenario.
    :param ego_vehicle: Ego obstacle, may contain both past and future trajectories.
    :param destination: Path to save output to.
    :param ego_type: Type of the ego vehicle (usually CAR).
    :param ego_harm: Harm for the ego vehicle.
    :param ego_v: Impact speed of the ego vehicle.
    :param ego_mass: Mass of the ego vehicle.
    :param obs_harm: Harm for the obstacle.
    :param obs_type: Type of obstacle.
    :param obs_v: Velocity of the obstacle.
    :param obs_mass: Estimated mass of the obstacle.
    :param pdof: Principle degree of force.
    :param ego_angle: Angle of impact area for the ego vehicle.
    :param obs_angle: Angle of impact area for the obstacle.
    :param time_step: Current time step.
    :param modes: Risk modes. Read from risk.json.
    :param planning_problem: Considered planning problem. Defaults to None.
    :param global_path: Global path for the planning problem. Defaults to None.
    :param driven_traj: Already driven trajectory of the ego vehicle. Defaults to None.
    """

    # clear everything
    plt.cla()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(5.833, 8.25)

    ego_params = DynamicObstacleParams()
    ego_params.time_begin = time_step
    ego_params.draw_icon = True
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
    ego_params.vehicle_shape.occupancy.shape.zorder = 50
    ego_params.vehicle_shape.occupancy.shape.opacity = 1

    # set plot limits to show the road section around the ego vehicle
    position = ego_vehicle.state_at_time(time_step).position
    plot_limits = [position[0] - 20,
                   position[0] + 20,
                   position[1] - 20,
                   position[1] + 20]

    scenario_params = MPDrawParams()
    scenario_params.time_begin = time_step
    scenario_params.dynamic_obstacle.show_label = True
    scenario_params.dynamic_obstacle.draw_icon = True
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#0065BD"
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 50
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 1

    rnd = MPRenderer(ax=ax2, plot_limits=plot_limits)
    # plot the scenario at the current time step
    scenario.draw(rnd, draw_params=scenario_params)

    plt.gca().set_aspect('equal')

    # draw the planning problem
    if planning_problem is not None:
        planning_problem.draw(rnd)

    # mark the ego vehicle
    if ego_vehicle is not None:
        ego_vehicle.draw(rnd, draw_params=ego_params)

    rnd.render()

    # Draw global path
    if global_path is not None:
        plt.plot(global_path[:, 0], global_path[:, 1], color='blue',
                 zorder=20, label='global path')

    # draw driven trajectory
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        plt.plot(x, y, color='green', zorder=25)

    # get the target time to show it in the title
    if hasattr(planning_problem.goal.state_list[0], 'time_step'):
        target_time_string = ('Target-time: %.1f s - %.1f s' %
                              (planning_problem.goal.state_list[0].
                               time_step.start * scenario.dt,
                               planning_problem.goal.state_list[0].
                               time_step.end * scenario.dt))
    else:
        target_time_string = 'No target-time'

    plt.legend()
    plt.title('Time: {0:.1f} s'.format(time_step * scenario.dt) + '    ' +
              target_time_string)

    # get mode description
    if modes["harm_mode"] == "log_reg":
        mode = "logistic regression"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "ref_speed":
        mode = "reference speed"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "gidas":
        mode = "GIDAS P(MAIS 2+)"
        harm = "P(MAIS 2+)"
    else:
        mode = "None"
        harm = "None"

    # get angle mode
    if modes["ignore_angle"] is True or modes["harm_mode"] == "gidas":
        angle = "ignoring impact areas"
    else:
        if modes["reduced_angle_areas"]:
            angle = "considering impact areas reduced on front, side, and " \
                    "rear crashes"
        else:
            angle = "considering impact areas according to the clock system"

        if modes["sym_angle"]:
            angle += " with symmetric coefficients"
        else:
            angle += " with asymmetric coefficients"

    # description of crash
    description = "Collision at {:.1f} s in ". \
                      format(time_step * scenario.dt) + \
                  str(scenario.scenario_id) + \
                  "\n\nCalculate harm using the " + mode + " model by " + angle + \
                  "\n\nEgo vehicle harm " + harm + ": {:.3f}".format(ego_harm) + \
                  "\nObstacle harm " + harm + ": {:.3f}".format(obs_harm) + \
                  "\n\nCrash parameters:\n\nEgo type: " + str(ego_type)[13:] + \
                  "\nEgo velocity: {:.2f}m/s". \
                      format(ego_v) + \
                  "\nEgo mass: {:.0f}kg".format(ego_mass) + \
                  "\nImpact angle for the ego vehicle: {:.2f}°". \
                      format(ego_angle * 180 / np.pi) + \
                  "\n\nObstacle type: " + str(obs_type)[13:]
    if obs_mass is not None and obs_v is not None and obs_angle is not None:
        description += "\nObstacle velocity: {:.2f}m/s".format(obs_v) + \
                       "\nObstacle mass: {:.0f}kg".format(obs_mass) + \
                       "\nImpact angle for the obstacle: {:.2f}°". \
                           format(obs_angle * 180 / np.pi)
    description += "\n\nPrinciple degree of force: {:.2f}°". \
        format(pdof * 180 / np.pi)

    # add description of crash
    ax1.axis('off')
    ax1.text(0, 1, description, verticalalignment='top', fontsize=8,
             wrap=True)

    fig.suptitle("Collision in " + str(scenario.scenario_id) + " detected",
                 fontsize=16)

    # Create directory for pictures
    destination = os.path.join(destination, "collisions")
    if not os.path.exists(destination):
        os.makedirs(destination)

    plt.savefig(destination + "/" + str(scenario.scenario_id) + ".svg", format="svg")
    plt.close()


def plot_final_trajectory(scenario: Scenario, planning_problem: PlanningProblem, state_list: List[CustomState],
                          config: Configuration, log_path: str, ref_path: np.ndarray = None, save=None, show=None):
    """
    Function plots occupancies for a given CommonRoad trajectory (of the ego vehicle)
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param state_list: List of trajectory States
    :param config: Configuration object for plot/save settings
    :param ref_path: Reference path as [(nx2) np.ndarray] (optional)
    :param save_path: Path to save plot as .png (optional)
    """
    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10))

    # set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
    rnd.draw_params.planning_problem.initial_state.state.radius = 0.5

    # set occupancy shape params
    occ_params = ShapeParams()
    occ_params.facecolor = '#E37222'
    occ_params.edgecolor = '#9C4100'
    occ_params.opacity = 1.0
    occ_params.zorder = 51

    # visualize scenario
    scenario.draw(rnd)
    # visualize planning problem
    planning_problem.draw(rnd)
    # visualize occupancies of trajectory
    for i in range(len(state_list)):
        state = state_list[i]
        occ_pos = Rectangle(length=config.vehicle.length, width=config.vehicle.width, center=state.position,
                            orientation=state.orientation)
        if i >= 1:
            occ_params.opacity = 0.3
            occ_params.zorder = 50
        occ_pos.draw(rnd, draw_params=occ_params)
    # render scenario and occupancies
    rnd.render()

    # visualize trajectory
    pos = np.asarray([state.position for state in state_list])
    rnd.ax.plot(pos[:, 0], pos[:, 1], color='k', marker='x', markersize=3.0, markeredgewidth=0.4, zorder=21,
                linewidth=0.8)

    # visualize reference path
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                    label='reference path')

    # save as .png file
    if config.visualization.save_plots or save:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{scenario.scenario_id}_final_trajectory.svg", format='svg',
                    bbox_inches='tight')

    # show plot
    if config.visualization.show_plots or show:
        matplotlib.use("TkAgg")
        # plt.show(block=False)
        # plt.pause(0.0001)


def plot_multiagent_scenario_final_trajectories(scenario, agent_list,
                                       config,
                                       log_path,
                                       show, save):
    """
    Function plots occupancies for a given CommonRoad trajectory (of the ego vehicle)
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param state_list: List of trajectory States
    :param config: Configuration object for plot/save settings
    :param ref_path: Reference path as [(nx2) np.ndarray] (optional)
    :param save_path: Path to save plot as .png (optional)
    """
    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10))

    # set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
    rnd.draw_params.planning_problem.initial_state.state.radius = 0.5

    # set occupancy shape params
    occ_params = ShapeParams()
    occ_params.facecolor = '#E37222'
    occ_params.edgecolor = '#9C4100'
    occ_params.opacity = 1.0
    occ_params.zorder = 51

    # visualize scenario
    scenario.draw(rnd)
    for agent in agent_list:
        # visualize planning problem
        agent.planning_problem.draw(rnd)
        # visualize occupancies of trajectory
        for i, state in enumerate(agent.record_state_list):
            # state = agent.record_state_list[i]
            occ_pos = Rectangle(length=config.vehicle.length, width=config.vehicle.width, center=state.position,
                                orientation=state.orientation)
            if i >= 1:
                occ_params.opacity = 0.3
                occ_params.zorder = 50
            occ_pos.draw(rnd, draw_params=occ_params)

    # render scenario and occupancies
    rnd.render()

    for agent in agent_list:
        # visualize trajectory
        pos = np.asarray([state.position for state in agent.record_state_list])
        rnd.ax.plot(pos[:, 0], pos[:, 1], color='k', marker='x', markersize=3.0, markeredgewidth=0.4, zorder=21,
                    linewidth=0.8)
        # visualize reference path
        if agent.reference_path is not None:
            rnd.ax.plot(agent.reference_path[:, 0], agent.reference_path[:, 1], color='g', marker='.', markersize=1, zorder=19,
                            linewidth=0.8,
                            label='reference path')

    # save as .png file
    if save:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{scenario.scenario_id}_final_trajectory.svg", format='svg',
                    bbox_inches='tight')

    # show plot
    if show:
        matplotlib.use("TkAgg")
        # plt.show(block=False)
        # plt.pause(0.0001)


def visualize_collision_checker(scenario: Scenario, cc: pycrcc.CollisionChecker):
    """
    Visualizes the collision checker, i.e., all collision objects and, if applicable, the road boundary.
    :param scenario CommonRoad scenario object
    :param cc pycrcc.CollisionChecker object
    """
    rnd = MPRenderer(figsize=(20, 10))
    scenario.lanelet_network.draw(rnd)
    cc.draw(rnd)
    rnd.render(show=True)


def visualize_scenario_and_pp(scenario: Scenario, planning_problem: PlanningProblem=None, save_path=None, cosy=None, t = 0):
    """Visualizes scenario, planning problem and (optionally) the reference path"""
    plot_limits = None
    ref_path = None
    if cosy is not None:
        try:
            ref_path = cosy.reference
        except:
            ref_path = np.array(cosy.reference_path())
        # x_min = np.min(ref_path[:, 0]) - 150
        # x_max = np.max(ref_path[:, 0]) + 150
        # y_min = np.min(ref_path[:, 1]) - 150
        # y_max = np.max(ref_path[:, 1]) + 150
        # plot_limits = [x_min, x_max, y_min, y_max]
        plot_limits = None

    rnd = MPRenderer(figsize=(20, 10), plot_limits=plot_limits)
    rnd.draw_params.time_begin = t
    rnd.draw_params.traffic_light.draw_traffic_lights = False
    rnd.draw_params.traffic_sign.draw_traffic_signs = False
    scenario.draw(rnd)
    if planning_problem is not None:
        planning_problem.draw(rnd)
    rnd.render()
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=20,
                    linewidth=0.8, label='reference path')
        if hasattr(cosy, "ccosy"):
            proj_domain_border = np.array(cosy.ccosy.projection_domain())
        else:
            proj_domain_border = np.array(cosy.projection_domain())
        rnd.ax.plot(proj_domain_border[:, 0], proj_domain_border[:, 1], color="orange", linewidth=0.8)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "reference_path.svg"), format='svg', bbox_inches='tight')
    else:
        plt.show()


def green_to_red_colormap():
    """Define a colormap that fades from green to red."""
    # This dictionary defines the colormap
    cdict = {
        "red": (
            (0.0, 0.0, 0.0),  # no red at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.8, 0.8),
        ),  # set to 0.8 so its not too bright at 1
        "green": (
            (0.0, 0.8, 0.8),  # set to 0.8 so its not too bright at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no green at 1
        "blue": (
            (0.0, 0.0, 0.0),  # no blue at 0
            (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
            (1.0, 0.0, 0.0),
        ),  # no blue at 1
    }

    # Create the colormap using the dictionary
    GnRd = colors.LinearSegmentedColormap("GnRd", cdict)

    return GnRd
