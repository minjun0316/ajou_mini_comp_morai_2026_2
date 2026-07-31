#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import time
import rospy
import rospkg
from math import cos,sin,pi,sqrt,pow,atan2
from geometry_msgs.msg import Point,PoseWithCovarianceStamped
from nav_msgs.msg import Odometry,Path
from morai_msgs.msg import CtrlCmd,EgoVehicleStatus,GetTrafficLightStatus
import numpy as np
import tf
from tf.transformations import euler_from_quaternion,quaternion_from_euler

# advanced_purepursuit 은 차량의 차량의 종 횡 방향 제어 예제입니다.
# Purpusuit 알고리즘의 Look Ahead Distance 값을 속도에 비례하여 가변 값으로 만들어 횡 방향 주행 성능을 올립니다.
# 횡방향 제어 입력은 주행할 Local Path (지역경로) 와 차량의 상태 정보 Odometry 를 받아 차량을 제어 합니다.
# 종방향 제어 입력은 목표 속도를 지정 한뒤 목표 속도에 도달하기 위한 Throttle control 을 합니다.
# 종방향 제어 입력은 longlCmdType 1(Throttle control) 이용합니다.

# 노드 실행 순서 
# 1. subscriber, publisher 선언
# 2. 속도 비례 Look Ahead Distance 값 설정
# 3. 좌표 변환 행렬 생성
# 4. Steering 각도 계산
# 5. PID 제어 생성
# 6. 도로의 곡률 계산
# 7. 곡률 기반 속도 계획
# 8. 제어입력 메세지 Publish

class pure_pursuit :
    def __init__(self):
        rospy.init_node('pure_pursuit', anonymous=True)

        #TODO: (1) subscriber, publisher 선언
        rospy.Subscriber("/global_path", Path, self.global_path_callback)
        rospy.Subscriber("/lattice_path", Path, self.path_callback)
        
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/Ego_topic",EgoVehicleStatus, self.status_callback)
        rospy.Subscriber("/GetTrafficLightStatus", GetTrafficLightStatus, self.traffic_light_callback)
        self.ctrl_cmd_pub = rospy.Publisher('ctrl_cmd',CtrlCmd, queue_size=1)

        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 1

        self.is_path = False
        self.is_odom = False 
        self.is_status = False
        self.is_global_path = False
        self.traffic_light_states = {}
        self.active_traffic_stop_id = None

        # 신호등별 ID, 정지 구역, 통과 신호 비트를 launch 파일에서 설정합니다.
        # 목록이 비어 있으면 신호등 제어가 비활성화되어 기존 주행에 영향을 주지 않습니다.
        self.traffic_stop_zones = rospy.get_param('~traffic_stop_zones', [])
        self.traffic_approach_distance = rospy.get_param('~traffic_approach_distance', 5.0)
        self.traffic_approach_velocity = rospy.get_param('~traffic_approach_velocity', 10.0)

        self.is_look_forward_point = False

        self.forward_point = Point()
        self.current_postion = Point()

        self.vehicle_length = 3.0
        self.lfd = 8
        self.min_lfd = 5
        self.max_lfd = 30
        self.lfd_gain = 0.78
        self.target_velocity = 25.0

        # --- 조향 정책 반영을 위한 파라미터 추가 ---
        self.max_steer_deg = 40.0  # 차량의 최대 조향각 (40도)
        self.max_steering_angle = self.max_steer_deg * pi / 180 
        # ------------------------------------------

        self.pid = pidControl()
        self.vel_planning = velocityPlanning(self.target_velocity/3.6, 0.15)
        while True:
            if self.is_global_path == True:
                self.velocity_list = self.vel_planning.curvedBaseVelocity(self.global_path, 50)
                break
            else:
                rospy.loginfo('Waiting global path data')

        rate = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():

            if self.is_path == True and self.is_odom == True and self.is_status == True:
                prev_time = time.time()

                self.current_waypoint = self.get_current_waypoint(self.status_msg,self.global_path)
                self.target_velocity = self.velocity_list[self.current_waypoint]*3.6

                # 신호등 정지 구역 5m 전부터 목표속도를 낮춰 급정지를 방지합니다.
                if self.is_near_traffic_stop_zone():
                    self.target_velocity = min(
                        self.target_velocity,
                        self.traffic_approach_velocity
                    )

                front_steer = self.calc_pure_pursuit()
                if self.is_look_forward_point :
                    self.ctrl_cmd_msg.front_steer = front_steer
                else : 
                    rospy.loginfo("no found forward point")
                    self.ctrl_cmd_msg.front_steer = 0.0
                
                output = self.pid.pid(self.target_velocity,self.status_msg.velocity.x*3.6)

                if output > 0.0:
                    self.ctrl_cmd_msg.accel = output
                    self.ctrl_cmd_msg.brake = 0.0
                else:
                    self.ctrl_cmd_msg.accel = 0.0
                    self.ctrl_cmd_msg.brake = -output

                # 현재 정지 구역에 대응하는 신호등의 허용 신호가 아니면 정지합니다.
                stop_zone = self.get_traffic_control_zone()
                if self.should_stop_for_traffic_light(stop_zone):
                    self.ctrl_cmd_msg.accel = 0.0
                    self.ctrl_cmd_msg.brake = 1.0
                    rospy.logwarn_throttle(
                        1.0,
                        "Traffic stop: id=%s, status=%s, allowed_bit=%s",
                        stop_zone['id'],
                        self.traffic_light_states.get(stop_zone['id'], -1),
                        stop_zone['allowed_signal']
                    )

                #TODO: (8) 제어입력 메세지 Publish
                # print(f"Target Vel: {self.target_velocity:.1f} | Final Steer: {front_steer:.4f}") # 디버깅용 출력 변경 가능
                self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)
                
            self.is_path = self.is_odom = self.is_status = False
            rate.sleep()

    def path_callback(self,msg):
        self.is_path=True
        self.path=msg  

    def odom_callback(self,msg):
        self.is_odom=True
        odom_quaternion=(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        _,_,self.vehicle_yaw=euler_from_quaternion(odom_quaternion)
        self.current_postion.x=msg.pose.pose.position.x
        self.current_postion.y=msg.pose.pose.position.y

    def status_callback(self,msg): ## Vehicl Status Subscriber 
        self.is_status=True
        self.status_msg=msg

    def traffic_light_callback(self, msg):
        self.traffic_light_states[msg.trafficLightIndex] = msg.trafficLightStatus

    def get_current_traffic_stop_zone(self):
        vehicle_x = self.status_msg.position.x
        vehicle_y = self.status_msg.position.y
        vehicle_z = self.status_msg.position.z

        for stop_zone in self.traffic_stop_zones:
            try:
                stop_zone['id'] = str(stop_zone['id'])
                stop_zone['allowed_signal'] = int(stop_zone['allowed_signal'])
                x_min = float(stop_zone['x_min'])
                x_max = float(stop_zone['x_max'])
                y_min = float(stop_zone['y_min'])
                y_max = float(stop_zone['y_max'])
                z_min = float(stop_zone['z_min'])
                z_max = float(stop_zone['z_max'])
            except (KeyError, TypeError, ValueError):
                rospy.logwarn_throttle(
                    5.0,
                    "Invalid traffic_stop_zones entry: %s",
                    stop_zone
                )
                continue

            is_in_zone = (
                x_min <= vehicle_x <= x_max
                and y_min <= vehicle_y <= y_max
                and z_min <= vehicle_z <= z_max
            )
            if is_in_zone:
                return stop_zone

        return None

    def get_traffic_control_zone(self):
        current_zone = self.get_current_traffic_stop_zone()
        if current_zone is not None:
            return current_zone

        # 빨간불에 한 번 정지한 뒤에는 차량이 정지 좌표 밖으로 조금 밀려도
        # 해당 신호가 허용 상태로 바뀔 때까지 같은 신호등 제어를 유지합니다.
        if self.active_traffic_stop_id is not None:
            for stop_zone in self.traffic_stop_zones:
                if str(stop_zone.get('id', '')) == self.active_traffic_stop_id:
                    return stop_zone

        return None

    def is_near_traffic_stop_zone(self):
        vehicle_x = self.status_msg.position.x
        vehicle_y = self.status_msg.position.y
        vehicle_z = self.status_msg.position.z

        for stop_zone in self.traffic_stop_zones:
            try:
                x_min = float(stop_zone['x_min'])
                x_max = float(stop_zone['x_max'])
                y_min = float(stop_zone['y_min'])
                y_max = float(stop_zone['y_max'])
                z_min = float(stop_zone['z_min'])
                z_max = float(stop_zone['z_max'])
            except (KeyError, TypeError, ValueError):
                continue

            # 차량과 직육면체 정지 구역 사이의 최단거리를 계산합니다.
            dx = max(x_min - vehicle_x, 0.0, vehicle_x - x_max)
            dy = max(y_min - vehicle_y, 0.0, vehicle_y - y_max)
            dz = max(z_min - vehicle_z, 0.0, vehicle_z - z_max)
            distance = sqrt(dx * dx + dy * dy + dz * dz)

            if distance <= self.traffic_approach_distance:
                return True

        return False

    def should_stop_for_traffic_light(self, stop_zone):
        if stop_zone is None:
            return False

        # -1(default)은 Python 비트 연산상 모든 비트가 켜진 것처럼 보이므로
        # 유효한 상태인지 먼저 검사합니다. 미수신/default 상태에서는 안전 정지합니다.
        traffic_light_status = self.traffic_light_states.get(stop_zone['id'], -1)
        if traffic_light_status < 0:
            self.active_traffic_stop_id = stop_zone['id']
            return True

        is_allowed_signal_on = (traffic_light_status & stop_zone['allowed_signal']) != 0
        if is_allowed_signal_on:
            self.active_traffic_stop_id = None
            return False

        self.active_traffic_stop_id = stop_zone['id']
        return True
        
    def global_path_callback(self,msg):
        self.global_path = msg
        self.is_global_path = True
    
    def get_current_waypoint(self,ego_status,global_path):
        min_dist = float('inf')        
        currnet_waypoint = -1
        for i,pose in enumerate(global_path.poses):
            dx = ego_status.position.x - pose.pose.position.x
            dy = ego_status.position.y - pose.pose.position.y

            dist = sqrt(pow(dx,2)+pow(dy,2))
            if min_dist > dist :
                min_dist = dist
                currnet_waypoint = i
        return currnet_waypoint

    def calc_pure_pursuit(self,):

        #TODO: (2) 속도 비례 Look Ahead Distance 값 설정
        self.lfd = (self.status_msg.velocity.x) * self.lfd_gain
        
        if self.lfd < self.min_lfd : 
            self.lfd=self.min_lfd
        elif self.lfd > self.max_lfd :
            self.lfd=self.max_lfd
        rospy.loginfo(self.lfd)
        
        vehicle_position=self.current_postion
        self.is_look_forward_point= False

        translation = [vehicle_position.x, vehicle_position.y]

        #TODO: (3) 좌표 변환 행렬 생성
        trans_matrix = np.array([
                [cos(self.vehicle_yaw), -sin(self.vehicle_yaw),translation[0]],
                [sin(self.vehicle_yaw),cos(self.vehicle_yaw),translation[1]],
                [0                    ,0                    ,1            ]])

        det_trans_matrix = np.linalg.inv(trans_matrix)

        for num,i in enumerate(self.path.poses) :
            path_point=i.pose.position

            global_path_point = [path_point.x,path_point.y,1]
            local_path_point = det_trans_matrix.dot(global_path_point)    

            if local_path_point[0]>0 :
                dis = sqrt(pow(local_path_point[0],2)+pow(local_path_point[1],2))
                if dis >= self.lfd :
                    self.forward_point = path_point
                    self.is_look_forward_point = True
                    break
        
        #TODO: (4) Steering 각도 계산
        theta = atan2(local_path_point[1],local_path_point[0])
        raw_steering_angle = atan2((2*self.vehicle_length*sin(theta)),self.lfd)

        # [동일 정책 반영] 방향 반전(-1 곱하기) 및 최대 조향각(40도) 기준 정규화
        normalized_steer = - (raw_steering_angle / self.max_steering_angle)
        front_steer = np.clip(normalized_steer, -1.0, 1.0)

        return front_steer

class pidControl:
    def __init__(self):
        self.p_gain = 0.3
        self.i_gain = 0.00
        self.d_gain = 0.01
        self.prev_error = 0
        self.i_control = 0
        self.controlTime = 0.02

    def pid(self,target_vel, current_vel):
        error = target_vel - current_vel

        #TODO: (5) PID 제어 생성
        p_control = self.p_gain * error
        self.i_control += self.i_gain * error * self.controlTime
        d_control = self.d_gain * (error-self.prev_error) / self.controlTime

        output = p_control + self.i_control + d_control
        self.prev_error = error

        return output

class velocityPlanning:
    def __init__ (self,car_max_speed, road_friciton):
        self.car_max_speed = car_max_speed
        self.road_friction = road_friciton

    def curvedBaseVelocity(self, gloabl_path, point_num):
        out_vel_plan = []

        for i in range(0,point_num):
            out_vel_plan.append(self.car_max_speed)

        for i in range(point_num, len(gloabl_path.poses) - point_num):
            x_list = []
            y_list = []
            for box in range(-point_num, point_num):
                x = gloabl_path.poses[i+box].pose.position.x
                y = gloabl_path.poses[i+box].pose.position.y
                x_list.append([-2*x, -2*y ,1])
                y_list.append((-x*x) - (y*y))

            #TODO: (6) 도로의 곡률 계산
            x_matrix = np.array(x_list)
            y_matrix = np.array(y_list)
            x_trans = x_matrix.T

            a_matrix = np.linalg.inv(x_trans.dot(x_matrix)).dot(x_trans).dot(y_matrix)
            a = a_matrix[0]
            b = a_matrix[1]
            c = a_matrix[2]
            r = sqrt(a*a+b*b-c)

            #TODO: (7) 곡률 기반 속도 계획
            v_max = sqrt(r*9.8*self.road_friction)

            if v_max > self.car_max_speed:
                v_max = self.car_max_speed
            out_vel_plan.append(v_max)

        for i in range(len(gloabl_path.poses) - point_num, len(gloabl_path.poses)-10):
            out_vel_plan.append(25.0 / 3.6)

        for i in range(len(gloabl_path.poses) - 10, len(gloabl_path.poses)):
            out_vel_plan.append(0)

        return out_vel_plan

if __name__ == '__main__':
    try:
        test_track=pure_pursuit()
    except rospy.ROSInterruptException:
        pass
