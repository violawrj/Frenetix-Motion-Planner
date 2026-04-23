#!/usr/bin/env python3
"""
Script to visualize CommonRoad XML scenarios without motion planning.
Just shows the scenario, road network, and recorded trajectories of dynamic obstacles.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional

from IPython.core.pylabtools import figsize

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams, ShapeParams
from commonroad.scenario.obstacle import DynamicObstacle


from commonroad.visualization.mp_renderer import ZOrders
# TODO: consider lowering the ZOrder of goal regions in the Frenetix plotter code
ZOrders.LANELET_LABEL = 25.0  # Or whatever z-order you want

# === Standard library ===
import os
import math
import shutil
import tempfile
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Callable, List, Optional, Set, Tuple, Union

# === Third-party libraries ===
import numpy as np
import shapely.geometry
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb
# Alternatively, remove this if not actually used in your code
# and import pathlib.Path instead when needed.

# === CommonRoad ===
import commonroad.geometry.shape  # only if you use other shapes besides Rectangle
from commonroad.geometry.shape import Rectangle
import commonroad.prediction.prediction  # only if used outside specific class imports
from commonroad.prediction.prediction import Occupancy, TrajectoryPrediction
import commonroad.scenario.obstacle  # same comment as above
from commonroad.scenario.obstacle import (
    DynamicObstacle,
    EnvironmentObstacle,
    Obstacle,
    PhantomObstacle,
    SignalState,
    StaticObstacle,
)
from commonroad.common.common_lanelet import LineMarking
from commonroad.common.util import Interval
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import TraceState
from commonroad.scenario.traffic_light import TrafficLight, TrafficLightState
from commonroad.scenario.traffic_sign import TrafficSign
from commonroad.scenario.trajectory import Trajectory

# === CommonRoad Visualization ===
from commonroad.visualization.draw_params import (
    BaseParam,
    DynamicObstacleParams,
    EnvironmentObstacleParams,
    InitialStateParams,
    LaneletNetworkParams,
    MPDrawParams,
    OccupancyParams,
    OptionalSpecificOrAllDrawParams,
    PhantomObstacleParams,
    PlanningProblemParams,
    PlanningProblemSetParams,
    ShapeParams,
    StateParams,
    StaticObstacleParams,
    TrafficLightParams,
    TrafficSignParams,
    TrajectoryParams,
    VehicleSignalParams,
)
from commonroad.visualization.drawable import IDrawable
from commonroad.visualization.icons import get_obstacle_icon_patch, supported_icons
from commonroad.visualization.renderer import IRenderer
from commonroad.visualization.traffic_sign import draw_traffic_light_signs
from commonroad.visualization.util import (
    LineCollectionDataUnits,
    LineDataUnits,
    approximate_bounding_box_dyn_obstacles,
    collect_center_line_colors,
    colormap_idx,
    get_arrow_path_at,
    get_tangent_angle,
    get_vehicle_direction_triangle,
    line_marking_to_linestyle,
    traffic_light_color_dict,
)


# Create subclass of the MPRenderer to allow for more flexible lanelet drawing
class CustomRenderer(MPRenderer):
    def draw_lanelet_network(
            self, obj: LaneletNetwork, draw_params: OptionalSpecificOrAllDrawParams[LaneletNetworkParams] = None
    ) -> None:
        from matplotlib.path import Path
        import matplotlib.collections as collections
        """
        Draws a lanelet network

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict that
            recreates the structure of an object,
        :return: None
        """
        if draw_params is None:
            draw_params = self.draw_params.lanelet_network
        elif isinstance(draw_params, MPDrawParams):
            draw_params = draw_params.lanelet_network

        traffic_lights = obj.traffic_lights
        traffic_signs = obj.traffic_signs
        intersections = obj.intersections
        lanelets = obj.lanelets

        time_begin = draw_params.time_begin
        if traffic_lights is not None:
            draw_traffic_lights = draw_params.traffic_light.draw_traffic_lights

            traffic_light_colors = draw_params.traffic_light
        else:
            draw_traffic_lights = False

        if traffic_signs is not None:
            draw_traffic_signs = draw_params.traffic_sign.draw_traffic_signs
            show_traffic_sign_label = draw_params.traffic_sign.show_label
        else:
            draw_traffic_signs = show_traffic_sign_label = False

        if intersections is not None and len(intersections) > 0:
            draw_intersections = draw_params.intersection.draw_intersections
        else:
            draw_intersections = False

        if draw_intersections is True:
            draw_incoming_lanelets = draw_params.intersection.draw_incoming_lanelets
            incoming_lanelets_color = draw_params.intersection.incoming_lanelets_color
            draw_crossings = draw_params.intersection.draw_crossings
            crossings_color = draw_params.intersection.crossings_color
            draw_successors = draw_params.intersection.draw_successors
            successors_left_color = draw_params.intersection.successors_left_color
            successors_straight_color = draw_params.intersection.successors_straight_color
            successors_right_color = draw_params.intersection.successors_right_color
            show_intersection_labels = draw_params.intersection.show_label
        else:
            draw_incoming_lanelets = draw_crossings = draw_successors = show_intersection_labels = False

        left_bound_color = draw_params.lanelet.left_bound_color
        right_bound_color = draw_params.lanelet.right_bound_color
        center_bound_color = draw_params.lanelet.center_bound_color
        unique_colors = draw_params.lanelet.unique_colors
        draw_stop_line = draw_params.lanelet.draw_stop_line
        stop_line_color = draw_params.lanelet.stop_line_color
        draw_line_markings = draw_params.lanelet.draw_line_markings
        show_label = draw_params.lanelet.show_label
        draw_border_vertices = draw_params.lanelet.draw_border_vertices
        draw_left_bound = draw_params.lanelet.draw_left_bound
        draw_right_bound = draw_params.lanelet.draw_right_bound
        draw_center_bound = draw_params.lanelet.draw_center_bound
        draw_start_and_direction = draw_params.lanelet.draw_start_and_direction
        draw_linewidth = draw_params.lanelet.draw_linewidth
        fill_lanelet = draw_params.lanelet.fill_lanelet
        facecolor = draw_params.lanelet.facecolor
        antialiased = draw_params.antialiased
        lanelet_zorder = draw_params.lanelet.zorder

        draw_lanlet_ids = draw_params.draw_ids

        colormap_tangent = draw_params.lanelet.colormap_tangent

        # Collect lanelets
        incoming_lanelets = set()
        incomings_left = {}
        incomings_id = {}
        crossings = set()
        all_successors = set()
        successors_left = set()
        successors_straight = set()
        successors_right = set()
        if draw_intersections:
            # collect incoming lanelets
            if draw_incoming_lanelets:
                incomings: List[set] = []
                inc_2_intersections = obj.map_inc_lanelets_to_intersections
                for intersection in intersections:
                    for incoming in intersection.incomings:
                        incomings.append(incoming.incoming_lanelets)
                        for l_id in incoming.incoming_lanelets:
                            incomings_left[l_id] = incoming.left_of
                            incomings_id[l_id] = incoming.incoming_id
                incoming_lanelets: Set[int] = set.union(*incomings)

            if draw_crossings:
                tmp_list: List[set] = [intersection.crossings for intersection in intersections]
                crossings: Set[int] = set.union(*tmp_list)

            if draw_successors:
                tmp_list: List[set] = [
                    incoming.successors_left for intersection in intersections for incoming in intersection.incomings
                ]
                successors_left: Set[int] = set.union(*tmp_list)
                tmp_list: List[set] = [
                    incoming.successors_straight
                    for intersection in intersections
                    for incoming in intersection.incomings
                ]
                successors_straight: Set[int] = set.union(*tmp_list)
                tmp_list: List[set] = [
                    incoming.successors_right for intersection in intersections for incoming in intersection.incomings
                ]
                successors_right: Set[int] = set.union(*tmp_list)
                all_successors = set.union(successors_straight, successors_right, successors_left)

        # select unique colors from colormap for each lanelet's center_line

        incoming_vertices_fill = list()
        crossing_vertices_fill = list()
        succ_left_paths = list()
        succ_straight_paths = list()
        succ_right_paths = list()

        vertices_fill = list()
        coordinates_left_border_vertices = []
        coordinates_right_border_vertices = []
        direction_list = list()
        center_paths = list()
        left_paths = list()
        right_paths = list()

        if draw_traffic_lights:
            center_line_color_dict = collect_center_line_colors(obj, traffic_lights, time_begin)

        cmap_lanelet = colormap_idx(len(lanelets))

        # collect paths for drawing
        for i_lanelet, lanelet in enumerate(lanelets):
            if isinstance(draw_lanlet_ids, list) and lanelet.lanelet_id not in draw_lanlet_ids:
                continue

            # project lanelet vertices to xy-plane as we make a 2D plot
            center_vertices_2d = lanelet.center_vertices[:, :2]
            left_vertices_2d = lanelet.left_vertices[:, :2]
            right_vertices_2d = lanelet.right_vertices[:, :2]

            def _draw_bound(vertices, line_marking, paths, coordinate_border_vertices):
                if draw_border_vertices:
                    coordinate_border_vertices.append(vertices)

                if (
                        draw_line_markings
                        and line_marking is not LineMarking.UNKNOWN
                        and line_marking is not LineMarking.NO_MARKING
                ):
                    linestyle, dashes, linewidth_metres = line_marking_to_linestyle(line_marking)
                    if lanelet.distance[-1] <= linewidth_metres:
                        paths.append(Path(right_vertices_2d, closed=False))
                    else:
                        tmp_vertices = vertices.copy()
                        line_string = shapely.geometry.LineString(tmp_vertices)
                        max_dist = line_string.project(shapely.geometry.Point(*vertices[-1])) - linewidth_metres / 2

                        if line_marking in (LineMarking.DASHED, LineMarking.BROAD_DASHED):
                            # In Germany, dashed lines are 6m long and 12m apart.
                            distances_start = np.arange(linewidth_metres / 2, max_dist, 12.0 + 6.0)
                            distances_end = distances_start + 6.0
                            # Cut off the last dash if it is too long.
                            distances_end[-1] = min(distances_end[-1], max_dist)
                            p_start = [line_string.interpolate(s).coords for s in distances_start]
                            p_end = [line_string.interpolate(s).coords for s in distances_end]
                            pts = np.squeeze(np.stack((p_start, p_end), axis=1), axis=2)
                            collection = LineCollectionDataUnits(
                                pts,
                                zorder=ZOrders.RIGHT_BOUND,
                                linewidth=linewidth_metres,
                                alpha=1.0,
                                color=right_bound_color,
                            )
                            self.static_collections.append(collection)
                        else:
                            # Offset, start and end of the line marking, to make them aligned with the lanelet.
                            tmp_vertices[0, :] = line_string.interpolate(linewidth_metres / 2).coords
                            tmp_vertices[-1, :] = line_string.interpolate(max_dist).coords
                            self.static_artists.append(
                                LineDataUnits(
                                    tmp_vertices[:, 0],
                                    tmp_vertices[:, 1],
                                    zorder=ZOrders.RIGHT_BOUND,
                                    linewidth=linewidth_metres,
                                    alpha=1.0,
                                    color=right_bound_color,
                                    linestyle=linestyle,
                                    dashes=dashes,
                                )
                            )
                else:
                    paths.append(Path(vertices, closed=False))

            # left bound
            if (draw_border_vertices or draw_left_bound) and (
                    lanelet.adj_left is None or not lanelet.adj_left_same_direction
            ):
                _draw_bound(
                    left_vertices_2d, lanelet.line_marking_left_vertices, left_paths, coordinates_left_border_vertices
                )

            # right bound
            if draw_border_vertices or draw_right_bound:
                _draw_bound(
                    right_vertices_2d,
                    lanelet.line_marking_right_vertices,
                    right_paths,
                    coordinates_right_border_vertices,
                )

            # stop line
            if draw_stop_line and lanelet.stop_line:
                # project stop line to xy-plane for 2D plot
                stop_line = np.vstack([lanelet.stop_line.start[:2], lanelet.stop_line.end[:2]])
                linestyle, dashes, linewidth_metres = line_marking_to_linestyle(lanelet.stop_line.line_marking)
                # cut off in the beginning, because linewidth_metres is added
                # later
                vec = stop_line[1, :] - stop_line[0, :]
                tangent = vec / np.linalg.norm(vec)
                stop_line[0, :] += linewidth_metres * tangent / 2
                stop_line[1, :] -= linewidth_metres * tangent / 2
                line = LineDataUnits(
                    stop_line[:, 0],
                    stop_line[:, 1],
                    zorder=ZOrders.STOP_LINE,
                    linewidth=linewidth_metres,
                    alpha=1.0,
                    color=stop_line_color,
                    linestyle=linestyle,
                    dashes=dashes,
                )
                self.static_artists.append(line)

            if unique_colors:
                # set center bound color to unique value
                center_bound_color = cmap_lanelet(i_lanelet)

            # direction arrow
            if draw_start_and_direction:
                center = center_vertices_2d[0]
                orientation = math.atan2(*(center_vertices_2d[1] - center)[::-1])
                lanelet_width = np.linalg.norm(right_vertices_2d[0] - left_vertices_2d[0])
                arrow_width = min(lanelet_width, 1.5)
                path = get_arrow_path_at(*center, orientation, arrow_width)
                if unique_colors:
                    direction_list.append(
                        matplotlib.patches.PathPatch(
                            path,
                            color=center_bound_color,
                            lw=0.5,
                            zorder=ZOrders.DIRECTION_ARROW,
                            antialiased=antialiased,
                        )
                    )
                else:
                    direction_list.append(path)

            # visualize traffic light state through colored center bound
            has_traffic_light = draw_traffic_lights and lanelet.lanelet_id in center_line_color_dict
            if has_traffic_light:
                light_state = center_line_color_dict[lanelet.lanelet_id]

                if light_state is not TrafficLightState.INACTIVE:
                    linewidth_metres = 0.75
                    # dashed line for red_yellow
                    linestyle = "--" if light_state == TrafficLightState.RED_YELLOW else "-"
                    dashes = (5, 5) if linestyle == "--" else (None, None)

                    # cut off in the beginning, because linewidth_metres is added later
                    tmp_center = center_vertices_2d.copy()
                    if lanelet.distance[-1] > linewidth_metres:
                        tmp_center[0, :] = lanelet.interpolate_position(linewidth_metres)[0][:2]
                    zorder = (
                        ZOrders.LIGHT_STATE_GREEN
                        if light_state == TrafficLightState.GREEN
                        else ZOrders.LIGHT_STATE_OTHER
                    )
                    line = LineDataUnits(
                        tmp_center[:, 0],
                        tmp_center[:, 1],
                        zorder=zorder,
                        linewidth=linewidth_metres,
                        alpha=0.7,
                        color=traffic_light_color_dict(light_state, traffic_light_colors),
                        linestyle=linestyle,
                        dashes=dashes,
                    )
                    self.dynamic_artists.append(line)

            # draw colored center bound. Hierarchy or colors: successors > usual
            # center bound
            is_successor = draw_intersections and draw_successors and lanelet.lanelet_id in all_successors
            if is_successor:
                if lanelet.lanelet_id in successors_left:
                    succ_left_paths.append(Path(center_vertices_2d, closed=False))
                elif lanelet.lanelet_id in successors_straight:
                    succ_straight_paths.append(Path(center_vertices_2d, closed=False))
                else:
                    succ_right_paths.append(Path(center_vertices_2d, closed=False))

            elif draw_center_bound:
                if unique_colors:
                    center_paths.append(
                        mpl.patches.PathPatch(
                            Path(center_vertices_2d, closed=False),
                            edgecolor=center_bound_color,
                            facecolor="none",
                            lw=draw_linewidth,
                            zorder=ZOrders.CENTER_BOUND,
                            antialiased=antialiased,
                        )
                    )
                elif colormap_tangent:
                    relative_angle = draw_params.relative_angle
                    points = center_vertices_2d.reshape(-1, 1, 2)
                    angles = get_tangent_angle(points[:, 0, :], relative_angle)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(0, 360)
                    lc = collections.LineCollection(
                        segments,
                        cmap="hsv",
                        norm=norm,
                        lw=draw_linewidth,
                        zorder=ZOrders.CENTER_BOUND,
                        antialiased=antialiased,
                    )
                    lc.set_array(angles)
                    self.static_collections.append(lc)

            is_incoming_lanelet = (
                    draw_intersections and draw_incoming_lanelets and (lanelet.lanelet_id in incoming_lanelets)
            )
            is_crossing = draw_intersections and draw_crossings and (lanelet.lanelet_id in crossings)

            # Draw lanelet area
            if fill_lanelet:
                if not is_incoming_lanelet and not is_crossing:
                    vertices_fill.append(np.concatenate((right_vertices_2d, np.flip(left_vertices_2d, 0))))

            # collect incoming lanelets in separate list for plotting in
            # different color
            if is_incoming_lanelet:
                incoming_vertices_fill.append(np.concatenate((right_vertices_2d, np.flip(left_vertices_2d, 0))))
            elif is_crossing:
                crossing_vertices_fill.append(np.concatenate((right_vertices_2d, np.flip(left_vertices_2d, 0))))

            # Draw labels
            if show_label or show_intersection_labels or draw_traffic_signs:
                strings = []
                if show_label:
                    strings.append(str(lanelet.lanelet_id))
                if is_incoming_lanelet and show_intersection_labels:
                    strings.append(f"int_id: {inc_2_intersections[lanelet.lanelet_id].intersection_id}")
                    strings.append("inc_id: " + str(incomings_id[lanelet.lanelet_id]))
                    strings.append("inc_left: " + str(incomings_left[lanelet.lanelet_id]))
                if draw_traffic_signs and show_traffic_sign_label:
                    traffic_signs_tmp = [obj._traffic_signs[id] for id in lanelet.traffic_signs]
                    if traffic_signs_tmp:
                        # add as text to label
                        str_tmp = "sign: "
                        add_str = ""
                        for sign in traffic_signs_tmp:
                            for el in sign.traffic_sign_elements:
                                # TrafficSignIDGermany(
                                # el.traffic_sign_element_id).name would give
                                # the
                                # name
                                str_tmp += add_str + el.traffic_sign_element_id.value
                                add_str = ", "

                        strings.append(str_tmp)

                label_string = ", ".join(strings)
                if len(label_string) > 0:
                    # compute normal angle of label box
                    clr_positions = lanelet.interpolate_position(0.5 * lanelet.distance[-1])
                    # project to xy-plane (last tuple element is an index, so we do not need to project it)
                    clr_positions = (clr_positions[0][:2], clr_positions[1][:2], clr_positions[2][:2], clr_positions[3])
                    normal_vector = np.array(clr_positions[1]) - np.array(clr_positions[2])
                    angle = np.rad2deg(math.atan2(normal_vector[1], normal_vector[0])) - 90
                    angle = angle if Interval(-90, 90).contains(angle) else angle - 180

                    # Custom modification for better view

                    # center_pos = np.array(clr_positions[0])
                    # left_pos = np.array(clr_positions[1])
                    # right_pos = np.array(clr_positions[2])

                    center_pos = np.array(clr_positions[0])  # label anchor point

                    # Get direction of travel (tangent) from centerline
                    centerline = lanelet.center_vertices
                    idx = clr_positions[3]

                    # Compute direction of travel (tangent)
                    if 0 < idx < len(centerline) - 1:
                        p_prev = np.array(centerline[idx - 1][:2])
                        p_next = np.array(centerline[idx + 1][:2])
                        tangent = p_next - p_prev
                        tangent /= np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else 1.0
                    else:
                        tangent = np.array([1.0, 0.0])  # fallback

                    # Compute normal vector pointing to the RIGHT of travel direction
                    normal = np.array([tangent[1], -tangent[0]])  # right-hand normal

                    # Apply shift
                    label_shift = 2.0  # meters; adjust this value as needed
                    offset_pos = center_pos + label_shift * normal
                    x, y = offset_pos[0], offset_pos[1]

                    # Unit vector from right to left
                    # normal = left_pos - right_pos
                    # normal /= np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else 1.0

                    # Pick a shift direction — you can use lanelet ID, left/right rule, or metadata
                    # Example 1: Push label slightly toward left boundary
                    # label_shift = 0.5  # meters; adjust as needed
                    # offset_pos = center_pos + label_shift * normal

                    # Now use this offset position for placing the label
                    # x, y = offset_pos[0], offset_pos[1]

                    self.static_artists.append(
                        matplotlib.text.Text(
                            # clr_positions[0][0],
                            # clr_positions[0][1],
                            offset_pos[0],
                            offset_pos[1],
                            label_string,
                            fontsize=5,
                            color="black",
                            alpha=0.75,
                            # bbox={"facecolor": center_bound_color, "pad": 2},
                            horizontalalignment="center",
                            verticalalignment="center",
                            rotation=angle,
                            zorder=ZOrders.LANELET_LABEL,
                        )
                    )

        # draw paths and collect axis handles
        if draw_right_bound:
            self.static_collections.append(
                collections.PathCollection(
                    right_paths,
                    edgecolor=right_bound_color,
                    facecolor="none",
                    lw=draw_linewidth,
                    zorder=lanelet_zorder + 0.1,
                    antialiased=antialiased,
                )
            )
        if draw_left_bound:
            self.static_collections.append(
                collections.PathCollection(
                    left_paths,
                    edgecolor=left_bound_color,
                    facecolor="none",
                    lw=draw_linewidth,
                    zorder=lanelet_zorder + 0.1,
                    antialiased=antialiased,
                )
            )
        if unique_colors:
            if draw_center_bound:
                if draw_center_bound:
                    self.static_collections.append(
                        collections.PatchCollection(
                            center_paths, match_original=True, zorder=ZOrders.CENTER_BOUND, antialiased=antialiased
                        )
                    )
                if draw_start_and_direction:
                    self.static_collections.append(
                        collections.PatchCollection(
                            direction_list, match_original=True, zorder=ZOrders.DIRECTION_ARROW, antialiased=antialiased
                        )
                    )

        elif not colormap_tangent:
            if draw_center_bound:
                self.static_collections.append(
                    collections.PathCollection(
                        center_paths,
                        edgecolor=center_bound_color,
                        facecolor="none",
                        lw=draw_linewidth,
                        zorder=ZOrders.CENTER_BOUND,
                        antialiased=antialiased,
                    )
                )
            if draw_start_and_direction:
                self.static_collections.append(
                    collections.PathCollection(
                        direction_list,
                        color=center_bound_color,
                        lw=0.5,
                        zorder=ZOrders.DIRECTION_ARROW,
                        antialiased=antialiased,
                    )
                )

        if successors_left:
            self.static_collections.append(
                collections.PathCollection(
                    succ_left_paths,
                    edgecolor=successors_left_color,
                    facecolor="none",
                    lw=draw_linewidth * 3.0,
                    zorder=ZOrders.SUCCESSORS,
                    antialiased=antialiased,
                )
            )
        if successors_straight:
            self.static_collections.append(
                collections.PathCollection(
                    succ_straight_paths,
                    edgecolor=successors_straight_color,
                    facecolor="none",
                    lw=draw_linewidth * 3.0,
                    zorder=ZOrders.SUCCESSORS,
                    antialiased=antialiased,
                )
            )
        if successors_right:
            self.static_collections.append(
                collections.PathCollection(
                    succ_right_paths,
                    edgecolor=successors_right_color,
                    facecolor="none",
                    lw=draw_linewidth * 3.0,
                    zorder=ZOrders.SUCCESSORS,
                    antialiased=antialiased,
                )
            )

        # fill lanelets with facecolor
        self.static_collections.append(
            collections.PolyCollection(
                vertices_fill,
                zorder=lanelet_zorder,
                facecolor=facecolor,
                edgecolor="none",
                antialiased=antialiased,
            )
        )
        if incoming_vertices_fill:
            self.static_collections.append(
                collections.PolyCollection(
                    incoming_vertices_fill,
                    facecolor=incoming_lanelets_color,
                    edgecolor="none",
                    zorder=ZOrders.INCOMING_POLY,
                    antialiased=antialiased,
                )
            )
        if crossing_vertices_fill:
            self.static_collections.append(
                collections.PolyCollection(
                    crossing_vertices_fill,
                    facecolor=crossings_color,
                    edgecolor="none",
                    zorder=ZOrders.CROSSING_POLY,
                    antialiased=antialiased,
                )
            )

        # draw_border_vertices
        if draw_border_vertices:
            coordinates_left_border_vertices = np.concatenate(coordinates_left_border_vertices, axis=0)
            # left vertices
            self.static_collections.append(
                collections.EllipseCollection(
                    np.ones([coordinates_left_border_vertices.shape[0], 1]) * 1.5,
                    np.ones([coordinates_left_border_vertices.shape[0], 1]) * 1.5,
                    np.zeros([coordinates_left_border_vertices.shape[0], 1]),
                    offsets=coordinates_left_border_vertices,
                    color=left_bound_color,
                    zorder=ZOrders.LEFT_BOUND + 0.1,
                )
            )

            coordinates_right_border_vertices = np.concatenate(coordinates_right_border_vertices, axis=0)
            # right_vertices
            self.static_collections.append(
                collections.EllipseCollection(
                    np.ones([coordinates_right_border_vertices.shape[0], 1]) * 1.5,
                    np.ones([coordinates_right_border_vertices.shape[0], 1]) * 1.5,
                    np.zeros([coordinates_right_border_vertices.shape[0], 1]),
                    offsets=coordinates_right_border_vertices,
                    color=right_bound_color,
                    zorder=ZOrders.LEFT_BOUND + 0.1,
                )
            )

        if draw_traffic_signs:
            # draw actual traffic sign
            for sign in traffic_signs:
                sign.draw(self, draw_params.traffic_sign)

        if draw_traffic_lights:
            # draw actual traffic light
            for light in traffic_lights:
                light.draw(self, draw_params.traffic_light)

def fix_trajectory_positions(scenario):
    """
    Fix trajectory position data format for prediction network compatibility.
    
    The WaleNet prediction network requires positions to be numpy arrays,
    but some SUMO-converted scenarios have positions stored as Python lists.
    This function ensures all positions are numpy arrays.
    """
    for obstacle in scenario.dynamic_obstacles:
        if hasattr(obstacle.prediction, 'trajectory') and obstacle.prediction.trajectory:
            for state in obstacle.prediction.trajectory.state_list:
                # Convert position to numpy array if it's a list
                if isinstance(state.position, list):
                    state.position = np.array(state.position)
    return scenario


def load_scenario(xml_file_path: str):
    """Load CommonRoad scenario from XML file."""
    try:
        scenario, planning_problem_set = CommonRoadFileReader(xml_file_path).open()
        # Fix trajectory position format for prediction network compatibility
        scenario = fix_trajectory_positions(scenario)
        return scenario, planning_problem_set
    except Exception as e:
        print(f"Error loading scenario: {e}")
        return None, None


def setup_visualization_params(timestep: int, show_agent_labels: bool = True,
                               show_lanelet_labels: bool = False, show_static_labels: bool = False):
    """Setup visualization parameters for clean rendering."""
    # Follow the same pattern as in cr_scenario_handler/utils/visualization.py

    # Main draw parameters
    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = True
    obs_params.dynamic_obstacle.show_label = show_agent_labels  # Configurable agent labels
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

    # Static obstacles - Gray (not red)
    obs_params.static_obstacle.show_label = show_static_labels  # Configurable static obstacle labels
    obs_params.static_obstacle.occupancy.shape.facecolor = "#808080"
    obs_params.static_obstacle.occupancy.shape.edgecolor = "#404040"

    # Enable traffic elements for realistic visualization
    obs_params.traffic_light.draw_traffic_lights = True
    obs_params.traffic_sign.draw_traffic_signs = True
    obs_params.lanelet_network.traffic_light.draw_traffic_lights = True
    obs_params.lanelet_network.traffic_sign.draw_traffic_signs = True
    obs_params.lanelet_network.traffic_light.show_label = False  # Don't clutter with IDs
    obs_params.lanelet_network.traffic_sign.show_label = False
    
    # Set time_begin for traffic lights to show correct state at each timestep
    obs_params.lanelet_network.traffic_light.time_begin = timestep
    obs_params.time_begin = timestep
    
    obs_params.lanelet_network.lanelet.show_label = show_lanelet_labels  # Configurable lanelet labels


    # FORCE disable ALL intersection drawing (same as working visualization.py)
    obs_params.lanelet_network.intersection.draw_intersections = False  # Disable entire intersection drawing
    obs_params.lanelet_network.intersection.draw_crossings = False  # Disable crossings specifically
    obs_params.lanelet_network.intersection.draw_incoming_lanelets = False
    obs_params.lanelet_network.intersection.draw_successors = False
    obs_params.lanelet_network.intersection.show_label = False

    obs_params.axis_visible = False

    return obs_params


def get_scenario_time_range(scenario):
    """Get the time range for the scenario based on dynamic obstacles."""
    max_time = 0
    min_time = float('inf')

    for obstacle in scenario.dynamic_obstacles:
        if obstacle.prediction is not None:
            max_time = max(max_time, obstacle.prediction.final_time_step)
        min_time = min(min_time, obstacle.initial_state.time_step)

    if min_time == float('inf'):
        min_time = 0

    return int(min_time), int(max_time)


def visualize_scenario_at_timestep(scenario, planning_problem_set, timestep: int,
                                   output_dir: str, scenario_name: str, show_plot: bool = False,
                                   show_agent_labels: bool = True, show_lanelet_labels: bool = False,
                                   show_static_labels: bool = False):

    # Configure matplotlib font sizes for labels
    # This controls the vehicle ID label size rendered by CommonRoad
    plt.rcParams['font.size'] = 8  # Smaller font size for vehicle IDs (default is 10)
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    rnd = CustomRenderer(figsize=(20, 10))
    obs_params = setup_visualization_params(timestep, show_agent_labels,
                                            show_lanelet_labels, show_static_labels)

    scenario.draw(rnd, draw_params=obs_params)

    if planning_problem_set is not None:
        for planning_problem in planning_problem_set.planning_problem_dict.values():
            planning_problem.draw(rnd)

    rnd.render()
    # Originally True and alpha=0.3
    rnd.ax.grid(False)

    # Only render grid for accurate coordinate clicks
    # Remove tick labels and ticks
    rnd.ax.set_xticks([])
    rnd.ax.set_yticks([])
    rnd.ax.tick_params(left=False, bottom=False)
    # Remove the axis spines (border lines)
    for spine in rnd.ax.spines.values():
        spine.set_visible(False)

    title = f"Scenario: {scenario.scenario_id} - Timestep: {timestep}"
    label_info = []
    if show_agent_labels:
        label_info.append("Agent IDs")
    if show_lanelet_labels:
        label_info.append("Lanelet IDs")
    if show_static_labels:
        label_info.append("Static IDs")
    if label_info:
        title += f" (Showing: {', '.join(label_info)})"

    output_file = os.path.join(output_dir, f"{scenario_name}_{timestep:03d}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_file}")

    if show_plot:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)
    else:
        plt.close()

def create_gif(scenario_name: str, image_dir: str, timesteps: range, output_dir: str, duration: float = 0.2):
    """Create animated GIF from saved images in `image_dir` and store it in `output_dir`.

    Uses the second frame (by order in the timesteps range) as the reference for image dimensions.
    """
    try:
        import imageio
        import os
        from PIL import Image
        import numpy as np

        images = []
        reference_shape = None
        buffer = {}  # Temporarily store first frame
        timestep_list = list(timesteps)

        if len(timestep_list) < 2:
            print("❌ At least two timesteps are required to determine a reference shape.")
            return

        reference_timestep = timestep_list[1]
        first_timestep = timestep_list[0]

        for t in timestep_list:
            img_path = os.path.join(image_dir, f"{scenario_name}_{t:03d}.png")
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                continue

            img = imageio.imread(img_path)

            if t == first_timestep:
                buffer[t] = img  # Buffer first frame
                continue

            if t == reference_timestep:
                reference_shape = img.shape[:2]
                print(f"📏 Reference shape set from frame {t:03d}: {reference_shape}")

                # Now process the buffered first frame
                first_img = buffer.pop(first_timestep, None)
                if first_img is not None:
                    if first_img.shape[:2] != reference_shape:
                        print(f"🔁 Resizing first frame {scenario_name}_{first_timestep:03d}.png from {first_img.shape[:2]} to {reference_shape}")
                        first_img = np.array(Image.fromarray(first_img).resize(reference_shape[::-1]))
                    images.append(first_img)

            if reference_shape:
                if img.shape[:2] != reference_shape:
                    print(f"⚠️  Frame {t:03d} shape {img.shape[:2]} doesn't match reference. Resizing...")
                    img = np.array(Image.fromarray(img).resize(reference_shape[::-1]))
                images.append(img)

        if images:
            gif_path = os.path.join(output_dir, f"{scenario_name}.gif")
            imageio.mimsave(gif_path, images, duration=duration, loop=0)
            print(f"✅ Created GIF: {gif_path}")
        else:
            print("⚠️  No images found for GIF creation")

    except ImportError:
        print("❌ imageio not installed. Install with: pip install imageio")
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")


# Uses temporary files to store the PNGs only for generating the GIF. Afterward, they are deleted.
def process_single_scenario(xml_file: str, base_output_dir: str, scenario_name: str,
                            show_plots: bool = False, create_animation: bool = True,
                            timestep_interval: int = 1, show_agent_labels: bool = False,
                            show_lanelet_labels: bool = False, show_static_labels: bool = False):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, xml_file)

    print(f"\n🎬 Processing: {xml_file}")
    print(f"Scenario name (from filename): {scenario_name}")
    print(f"Label settings: Agents={show_agent_labels}, Lanelets={show_lanelet_labels}, Static={show_static_labels}")

    scenario, planning_problem_set = load_scenario(xml_path)
    if scenario is None:
        print(f"❌ Failed to load scenario: {xml_file}")
        return False

    output_path = os.path.join(script_dir, base_output_dir)
    os.makedirs(output_path, exist_ok=True)

    print(f"Scenario ID (from file): {scenario.scenario_id}")
    print(f"Output: {output_path}")
    print(f"Dynamic obstacles: {len(scenario.dynamic_obstacles)}")
    print(f"Static obstacles: {len(scenario.static_obstacles)}")

    min_time, max_time = get_scenario_time_range(scenario)
    print(f"Time range: {min_time} to {max_time}")
    if max_time == 0:
        max_time = 0

    timesteps = range(min_time, max_time + 1, timestep_interval)
    print(f"Creating {len(timesteps)} visualizations...")

    with tempfile.TemporaryDirectory() as tmp_image_dir:
        for timestep in timesteps:
            if timestep % 10 == 0:
                print(f"  Timestep {timestep}...")
            visualize_scenario_at_timestep(
                scenario, planning_problem_set, timestep,
                tmp_image_dir, scenario_name, show_plots,
                show_agent_labels, show_lanelet_labels, show_static_labels
            )

        if create_animation and len(timesteps) > 1:
            print("Creating animated GIF...")
            create_gif(scenario_name, tmp_image_dir, timesteps, output_path)

    print(f"✅ Complete! GIF saved in: {output_path}")

    # TODO: remove, if coordinate stuff removed (thi smight not work with current way of vis)
    # === Store metadata for coordinate mapping ===
    dpi = 300
    figsize=(20, 10)
    dict_plot_limits = get_plot_limits(scenario)
    metadata = { # These parameters should be made sure to be globally shared (rn manual update is required)
        "dpi": dpi,
        "figsize": figsize,
        "plot_limits": dict_plot_limits[0],
    }
    json_name = str(scenario_name) + ".json"
    json_path = os.path.join(output_path, json_name)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return True

def get_plot_limits(scenario: Scenario):
    """
    The plot limits track the center of the ego vehicle.
    """

    def flatten(list_to_flat):
        return [item for sublist in list_to_flat for item in sublist]

    center_vertices = np.array(
        flatten(
            [lanelet.center_vertices for lanelet in scenario.lanelet_network.lanelets]
        )
    )

    min_coords = np.min(center_vertices, axis=0)
    max_coords = np.max(center_vertices, axis=0)
    dict_plot_limits = [
        [min_coords[0], max_coords[0], min_coords[1], max_coords[1]]
    ]

    return dict_plot_limits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Path to the input CommonRoad XML file")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    args = parser.parse_args()

    input_file = str(Path(args.input_file).resolve())
    output_dir = str(Path(args.output_dir).resolve())

    # Extract scenario name from the file name, without suffixes
    scenario_name = Path(input_file).stem

    # Configuration flags
    show_plots = False
    create_animation = True
    timestep_interval = 1
    show_agent_labels = True
    show_lanelet_labels = True
    show_static_labels = True

    success = process_single_scenario(
        input_file, output_dir, scenario_name, show_plots, create_animation,
        timestep_interval, show_agent_labels, show_lanelet_labels, show_static_labels
    )

    if success:
        print(f"\n🎉 All done!\n📂 Check the '{output_dir}' folder for your scenario visualizations!")
    else:
        print(f"\n❌ Processing failed!")


#TODO: don't forget to add coordinates for goal region choice
if __name__ == "__main__":
    main()

# source Frenetix-Motion-Planner/venv/bin/activate  python Frenetix-Motion-Planner/main_without_ego.py --input-file /home/avsaw1/sebastian/ChaBot7/Scenarios/DEU_Weimar-35_1_T-4/Modified/DEU_Weimar-35_1_T-4_M9328/DEU_Weimar-35_1_T-4_M9328.xml --output-dir /home/avsaw1/sebastian/ChaBot7/Scenarios/DEU_Weimar-35_1_T-4/Modified/DEU_Weimar-35_1_T-4_M9328
