#  Copyright 2020 Unity Technologies
#  Copyright 2026 gd-ros-tcp-connector contributors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Thin wrapper around ``rclpy.action.ActionClient`` that bridges the
ROS-TCP-Endpoint wire protocol to real ROS 2 Action servers.

A plain ``rclpy.create_client()`` cannot discover Action endpoints
because they use a different DDS endpoint kind / QoS profile than
regular services.  This module solves the problem by creating a real
``ActionClient`` and exposing ``send_goal`` / ``get_result`` /
``cancel_goal`` as methods the TCP server can call with serialized
CDR payloads.
"""

import re
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.serialization import deserialize_message, serialize_message

from .communication import RosSender


class RosActionClient(RosSender):
    """
    Manages one ``rclpy.action.ActionClient`` for a single action name.
    """

    def __init__(self, action_name, action_class, tcp_server):
        stripped = re.sub("[^A-Za-z0-9_]+", "", action_name)
        node_name = f"{stripped}_RosActionClient"
        RosSender.__init__(self, node_name)

        self.action_name = action_name
        self.action_class = action_class
        self.tcp_server = tcp_server

        self._action_client = ActionClient(self, action_class, action_name)
        self._goal_handles = {}  # goal_uuid_bytes -> ClientGoalHandle
        self._goal_handles_lock = threading.Lock()

    def send_goal(self, goal_data):
        """Send a goal. *goal_data* is CDR-serialized Goal body bytes.

        Returns the CDR-serialized SendGoal_Response (accepted + stamp),
        or None on failure.
        """
        goal_msg = deserialize_message(
            goal_data, self.action_class.Goal)

        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f"Action server {self.action_name} not available")
            return None

        # send_goal is synchronous (blocks until accepted/rejected).
        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._on_feedback)

        # Spin until the future completes.
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        goal_handle = future.result()
        if goal_handle is None:
            return None

        # Build a SendGoal_Response to send back to Godot.
        response_class = self.action_class.Impl.SendGoalService.Response
        resp = response_class()
        resp.accepted = goal_handle.accepted
        resp.stamp = goal_handle.stamp if hasattr(goal_handle, 'stamp') else resp.stamp

        if goal_handle.accepted:
            # Extract the 16-byte goal UUID from the goal_handle.
            goal_uuid = goal_handle.goal_id.uuid
            key = bytes(goal_uuid)
            with self._goal_handles_lock:
                self._goal_handles[key] = goal_handle

        return serialize_message(resp)

    def get_result(self, goal_id_data):
        """Request the result for a goal. *goal_id_data* is CDR-serialized
        GetResult_Request bytes (contains the goal UUID).

        Returns CDR-serialized GetResult_Response, or None on failure.
        """
        request_class = self.action_class.Impl.GetResultService.Request
        req = deserialize_message(goal_id_data, request_class)

        key = bytes(req.goal_id.uuid)
        with self._goal_handles_lock:
            goal_handle = self._goal_handles.get(key)

        if goal_handle is None:
            self.get_logger().error(
                f"get_result: no goal handle for UUID {key.hex()}")
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=300.0)
        result = result_future.result()

        # Clean up the handle.
        with self._goal_handles_lock:
            self._goal_handles.pop(key, None)

        if result is None:
            return None

        # Build GetResult_Response
        response_class = self.action_class.Impl.GetResultService.Response
        resp = response_class()
        resp.status = result.status
        resp.result = result.result
        return serialize_message(resp)

    def cancel_goal(self, goal_info_data):
        """Cancel a goal. *goal_info_data* is CDR-serialized
        CancelGoal_Request bytes.

        Returns CDR-serialized CancelGoal_Response, or None.
        """
        from action_msgs.srv import CancelGoal
        req = deserialize_message(goal_info_data, CancelGoal.Request)

        key = bytes(req.goal_info.goal_id.uuid)
        with self._goal_handles_lock:
            goal_handle = self._goal_handles.get(key)

        if goal_handle is None:
            self.get_logger().error(
                f"cancel_goal: no goal handle for UUID {key.hex()}")
            return None

        cancel_future = goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=10.0)
        cancel_response = cancel_future.result()

        if cancel_response is None:
            return None
        return serialize_message(cancel_response)

    def _on_feedback(self, feedback_msg):
        """Called by rclpy when the action server publishes feedback.

        Forward it to Godot over the TCP connection as a regular
        topic message on ``<action>/_action/feedback``.
        """
        # feedback_msg is Fibonacci.Impl.FeedbackMessage (the full wrapper
        # including goal_id). Serialize and send to Godot.
        self.tcp_server.send_unity_message(
            self.action_name + "/_action/feedback",
            feedback_msg.feedback)

    def unregister(self):
        self._action_client.destroy()
        self.destroy_node()
