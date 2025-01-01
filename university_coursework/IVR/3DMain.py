#!/usr/bin/env python2.7

import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import collections
import time
import os


class MainReacher():

    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')

        self.previousJointAngles = np.zeros(4)
        self.currentJointAngles = np.zeros(4)
        self.currentJointVelocities = np.zeros(4)

        self.max_vel = 2 # Used to calculate damping factor k in IK

        # Below variables are calculated in Jacobian method and needed in gravity compensation
        self.ee_local_y_axis = np.zeros(3)
        self.red_pos = np.zeros(3)
        self.green_pos = np.zeros(3)
        self.blue_pos = np.zeros(3)
        self.dark_blue_pos = np.zeros(3)

        self.env.reset()

    def detect_center_coordinates(self, xy_image, xz_image, rgb_color):
        """Returns: 3D coordinates of the center of the dot with color [rgb_color]
        by using both the xy and xz projections of the robot. """
        mask_xy = cv2.inRange(xy_image, rgb_color,rgb_color)  # Detect red color in xy image
        mask_xz = cv2.inRange(xz_image, rgb_color, rgb_color)  # and in xz image
        kernel = np.ones((5, 5), np.uint8)
        mask_xy = cv2.dilate(mask_xy, kernel, iterations=3) # Increase region of detected color
        mask_xz = cv2.dilate(mask_xz, kernel, iterations=3)
        M = cv2.moments(mask_xy) # Detect joint centers using moments
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        xy_coord = self.coordinate_convert((cX, cY))
        M = cv2.moments(mask_xz)
        cX = int(M["m10"] / M["m00"])
        cZ = int(M["m01"] / M["m00"])
        xz_coord = self.coordinate_convert((cX, cZ))
        # Get x,y coordinates from xy projection and the z coordinate from the xz projection.
        return np.array([xy_coord[0], xy_coord[1], xz_coord[1]])

    def detect_red(self, xy_image, xz_image):
        """Returns: 3D coordinates of the center of the red dot"""
        return self.detect_center_coordinates(xy_image, xz_image, (255, 0, 0))


    def detect_green(self, xy_image, xz_image):
        """Returns: 3D coordinates of the center of the green dot"""
        return self.detect_center_coordinates(xy_image, xz_image, (0, 255, 0))


    def detect_blue(self, xy_image, xz_image):
        """Returns: 3D coordinates of the center of the blue dot"""
        return self.detect_center_coordinates(xy_image, xz_image, (0, 0, 255))


    def detect_dark_blue(self, xy_image, xz_image):
        """Returns: 3D coordinates of the center of the dark blue dot"""
        return self.detect_center_coordinates(xy_image, xz_image, (0, 0, 127.5))

    def detect_valid_target_new(self, image):
        """ Detect valid target based on solidity. Solidity is given by the
        ratio between the contour area and the convex hull area. Assume valid
        target will always have lower solidity. """
        rgb_color = (178.5,178.5,178.5)
        mask = cv2.inRange(image, rgb_color,rgb_color)
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        readings = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            coord = self.coordinate_convert((x, y))
            readings.append((solidity, radius, coord))
        return min(readings)[-1] # Return coordinates of target with minimum solidity (= the valid target).

    def calculate_angles_and_velocities(self,xy,xz):
        ''' Calculates and sets the current joint angles and angular velocities.
            If an error occurs, the previous joint angles and velocities are kept
            and the error message is printed. '''
        try:
            self.currentJointAngles = self.detect_joint_angles(xy, xz)
            self.currentJointVelocities = self.angle_normalize(self.currentJointAngles - self.previousJointAngles) / self.env.dt
            self.previousJointAngles = self.currentJointAngles
        except Exception as e:
            print(e)


    def safe_arccos(self,point1,point2,point3):
        ''' Calculates the angle between the vectors point2-point1 and point3-point2.
        The value of the cosine is saturated to be between +1 and -1, to prevent
        errors due to rounding.'''
        v1 = point2-point1
        v2 = point3-point2

        cos = (v1.dot(v2)) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )

        if(cos > 1.0):
            cos=1.0
        elif(cos < -1.0):
            cos=-1.0

        return abs(np.arccos(cos))

    def detect_joint_angles(self, xy_image, xz_image):
        ''' Returns the joint angles (order: red,green,blue,dark_blue).'''
        # Detect the joints.
        red = self.detect_red(xy_image, xz_image)
        green = self.detect_green(xy_image, xz_image)
        blue = self.detect_blue(xy_image, xz_image)
        dark_blue = self.detect_dark_blue(xy_image, xz_image)

        # First joint angle:
        red_angle = math.atan2(red[2], red[0])

        # Rotation matrix for the first joint:
        rt_matrix_1 = self.rot_y_trans_x_matrix(red_angle, [0, 0, 0])

        # Obtain the local z axis for the first joint, by rotating the previous one
        # around -y.
        oz_after_first_rot = rt_matrix_1.dot(np.array([0, 0, 1, 1]))[0:3]

        # Calculate the second joint angle. Ref_vector (local z axis) allows us
        # to find the correct sign of the angle.
        ref_vector = oz_after_first_rot
        green_angle_unsigned = self.safe_arccos(np.array([0,0,0]),red,green) # get the magnitude of the angle
        cross_1 = np.cross(red, green - red) # cross between the first and the second link
        sign = cross_1.dot(ref_vector) # the dot product between the ref vector and the previous cross gives the correct sign

        # Second joint angle:
        green_angle = np.sign(sign) * green_angle_unsigned

        # We use the same ref_vector as for the green angle as the local Oz axis stays the same
        blue_angle_unsigned = self.safe_arccos(red,green,blue)

        cross_2 = np.cross(green-red, blue-green)
        sign = cross_2.dot(ref_vector)

        # Third joint angle:
        blue_angle = np.sign(sign) * blue_angle_unsigned

        # Rotation matrices around the local Oz axis (after the first rotation).
        rt_matrix_2 = self.rodrigues_rotation_matrix_safer(green_angle,oz_after_first_rot)
        rt_matrix_3 = self.rodrigues_rotation_matrix_safer(blue_angle,oz_after_first_rot)

        # Reference vector for the last rotation:
        ref_vector = rt_matrix_3.dot(rt_matrix_2.dot(np.array([0, 1, 0])))

        dark_blue_angle_unsigned = self.safe_arccos(green,blue,dark_blue)

        cross_3 = np.cross(blue - green, dark_blue - blue)
        sign = cross_3.dot(ref_vector)

        # Last joint angle (around local MINUS y axis):
        dark_blue_angle = (-1) * np.sign(sign) * dark_blue_angle_unsigned

        return np.array([red_angle, green_angle, blue_angle, dark_blue_angle])

    def transform_to_unit_vector(self, vector):
        '''Helper method to convert a vector to unit vector. (np illiterate)'''
        magnitude = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        if(magnitude < 1e-7):
            raise ValueError(
                "transform_to_unit_vector: provided vector has magnitude 0")
        else:
            return np.array([vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude])

    def rot_y_trans_x_matrix(self, angle, trans):
        '''Helper method which creates a homogeneous transformation matrix
        (rotation around y by [angle], translation by [trans]). '''
        cos = np.cos(angle)
        sin = np.sin(angle)
        return np.array([[cos, 0, -sin, trans[0]], [0, 1, 0, trans[1]], [sin, 0, cos, trans[2]], [0, 0, 0, 1]])

    # More "numerically sound" method to calculate the rotation matrix:
    def rodrigues_rotation_matrix_safer(self,angle,vector):
        ''' Calculates Rodrigues' rotation matrix around axis [vector] by [angle].
        It's safer, because it's more numerically sound (than what we previously tried).'''
        unit_vector = self.transform_to_unit_vector(vector)

        c = np.cos(angle)
        s = np.sin(angle)

        ux = unit_vector[0]
        uy = unit_vector[1]
        uz = unit_vector[2]

        W = np.array([[0,-uz,uy],[uz,0,-ux],[-uy,ux,0]])

        rot_matrix = np.eye(3) + s * W + (2*(np.sin(angle/2)**2)) * (W.dot(W))

        return rot_matrix


    def rodrigues_rot_homogen(self,angle,axis):
        ''' Adds a row and a column to the Rodrigues matrix so it can be used
        with homogeneous coordinates.'''
        rot_matrix = self.rodrigues_rotation_matrix_safer(angle,axis)
        rot_homogen = np.zeros((4,4))
        rot_homogen[0:3,0:3] = rot_matrix
        rot_homogen[3,3] = 1

        return rot_homogen

    def FK(self,joint_angles):
        ''' Returns the end effector position.'''
        translation = np.eye(4)
        translation [0,3] = 1

        j1_transform = self.rodrigues_rot_homogen(joint_angles[0],[0,-1,0]).dot(translation)
        j2_transform = self.rodrigues_rot_homogen(joint_angles[1],[0,0,1]).dot(translation)
        j3_transform = self.rodrigues_rot_homogen(joint_angles[2],[0,0,1]).dot(translation)
        j4_transform = self.rodrigues_rot_homogen(joint_angles[3],[0,-1,0]).dot(translation)

        dark_blue_pos = j1_transform.dot(j2_transform.dot(j3_transform.dot(j4_transform.dot([0,0,0,1]))))[0:3]

        return dark_blue_pos

    def Jacobian(self,joint_angles):
        ''' Calculates the Jacobian.'''
        jacobian = np.zeros((3,4))

        translation = np.eye(4)
        translation[0,3] = 1

        # Transforation matrices along the original axes of rotation.
        j1_transform = self.rodrigues_rot_homogen(joint_angles[0],[0,-1,0]).dot(translation)
        j2_transform = self.rodrigues_rot_homogen(joint_angles[1],[0,0,1]).dot(translation)
        j3_transform = self.rodrigues_rot_homogen(joint_angles[2],[0,0,1]).dot(translation)
        j4_transform = self.rodrigues_rot_homogen(joint_angles[3],[0,-1,0]).dot(translation)

        # Why do we multiply them like this? - That's saved for the presentation ;)
        # Because at the beginning, the local rotation axes for each joint are the same as the global axes.
        red_pos = j1_transform.dot([0,0,0,1])[0:3]
        green_pos = j1_transform.dot(j2_transform.dot([0,0,0,1]))[0:3]
        blue_pos = j1_transform.dot(j2_transform.dot(j3_transform.dot([0,0,0,1])))[0:3]
        dark_blue_pos = j1_transform.dot(j2_transform.dot(j3_transform.dot(j4_transform.dot([0,0,0,1]))))[0:3]

        # Save the joint positions for the gravity compensation part.
        self.red_pos = red_pos
        self.green_pos = green_pos
        self.blue_pos = blue_pos
        self.dark_blue_pos = dark_blue_pos

        # Getting the rotation axes (same way as in the method we get the angles):

        # Rotation matrix for the first rotation:
        rt_matrix_1 = self.rot_y_trans_x_matrix(joint_angles[0], [0, 0, 0])

        # Rotate a vector parallel to the z-axis around the y-axis.
        first_and_second_axis = rt_matrix_1.dot(np.array([0, 0, 1, 1]))[0:3]

        # Rotation matrices around the Oz axis after the first rotation.
        rt_matrix_2 = self.rodrigues_rotation_matrix_safer(joint_angles[1],first_and_second_axis)
        rt_matrix_3 = self.rodrigues_rotation_matrix_safer(joint_angles[2],first_and_second_axis)

        # Reference vector for the last rotation:
        third_axis = rt_matrix_3.dot(rt_matrix_2.dot(np.array([0, -1, 0])))

        first_and_second_axis = self.transform_to_unit_vector(first_and_second_axis)
        third_axis            = self.transform_to_unit_vector(third_axis)

        # Globally store axis of rotation of last joint
        self.ee_local_y_axis = third_axis
        self.first_and_second_axis = first_and_second_axis

        # Construct the Jacobian:
        jacobian[0:3,0] = np.cross(np.array([0,-1,0]),dark_blue_pos).T
        jacobian[0:3,1] = np.cross(first_and_second_axis,dark_blue_pos-red_pos).T
        jacobian[0:3,2] = np.cross(first_and_second_axis,dark_blue_pos-green_pos).T
        jacobian[0:3,3] = np.cross(third_axis,dark_blue_pos-blue_pos).T

        return jacobian

    def IK(self,current_joint_angles,target):
        ''' Inverse Kinematics.'''
        # Get current end effector position.
        curr_pos = self.FK(current_joint_angles)
        pos_error = target-curr_pos

        J = self.Jacobian(current_joint_angles)

        # Inverse Kinematics using approximate pseudoinverse with Jacobian Transpose.
        '''
        J_transpose = np.transpose(J)
    	J_mult = np.matmul(J,J_transpose)
        if(np.linalg.matrix_rank(J_mult) < 3):
			J_inverse = J_transpose
			dq = J_inverse.dot(pos_error)
			return dq
        else:
			J_inverse = np.matmul(J_transpose,np.linalg.inv(J_mult))
			dq = J_inverse.dot(pos_error)
			return dq'''


        # Inverse Kinematics using the Damped Least Squares Inverse method.

        # SVD decomposition of our Jacobian:
        u,s,v = np.linalg.svd(J)

        # Select minimum value of s:https://www.facebook.com/
        s_min = s[s.shape[0]-1]

        # Calculate damping factor k, as specified here:
        # https://www.engr.colostate.edu/~aam/pdf/journals/4.pdf
        max_xi = -3000
        for i in range(u.shape[1]):
            xi = (u[:,i]).reshape((3,)).dot(pos_error)
            if(xi > max_xi):
                max_xi = xi
        k = max_xi/(2*self.max_vel)

        if(s_min > k):
            # Calculate damping factor:
            if(s_min > (1/self.max_vel)):
                k = 0
            else:
                k = np.sqrt((s_min/self.max_vel)-s_min**2)

        J_transpose = np.transpose(J)
        J_interm = np.linalg.inv(J.dot(J_transpose) + (k**2)*np.eye(3))
        J_star = J_transpose.dot(J_interm)

        dq = J_star.dot(pos_error)

        # Impose a hard limit on the angle velocities (the above method tries to
        # limit them but is not always successful) - no longer used here.

        return self.scale(7,dq)

        # Inverse Kinematics using the determinant of J*J.T (when it's close to
        # 0, inverse the s matrix manually).
        '''
        J_transpose = np.transpose(J)
        det = np.linalg.det(J.dot(J_transpose))
        if(abs(det) > 0.005**2):
            J_inverse = np.matmul(J_transpose,np.linalg.inv(J.dot(J_transpose)))
            dq = J_inverse.dot(pos_error)
            return self.scale(4,dq)
        else:
            u,s,v = np.linalg.svd(J.dot(J_transpose))
            for i in range(len(s)):
                if(s[i]<.005):
                    s[i] = 0
                else:
                    s[i] = 1.0/s[i]
            J_dash = np.dot(v,np.dot(np.diag(s),u.T))
            J_inverse = np.dot(J_transpose,J_dash)
            return self.scale(4,J_inverse.dot(pos_error))'''



    def scale(self, max_vel, velocities):
        ''' Impose a hard limit on the magnitude of the [velocities]. '''
        for i in range(len(velocities)):
            if(velocities[i] > max_vel):
                velocities[i] = max_vel
            elif(velocities[i] < -max_vel):
                velocities[i] = -max_vel
        return velocities

    def grav(self, current_joint_angles):
        ''' Calculates the torque caused by gravity for each joint.
            Used to perform gravity compensation in the PD controller.'''

        # Get the local axis of rotation for the end effector.
        ee_rotation_axis = self.ee_local_y_axis

        # Coordinates of the centers of mass of the links.
        center1 = (self.dark_blue_pos + self.blue_pos)*0.5
        center2 = (self.green_pos + self.blue_pos)*0.5
        center3 = (self.red_pos+self.green_pos)*0.5

        # torque_xy = torque caused by link x on joint y
        # Here, joint 2 = red, joint 3 = green, joint 4 = blue.
        # Reference: http://homepages.wmich.edu/~kamman/ME256MomentAboutanAxis.pdf.
        torque_42 = np.cross(center1-self.red_pos,np.array([0,+9.8,0])).dot(self.first_and_second_axis)
        torque_32 = np.cross(center2-self.red_pos,np.array([0,+9.8,0])).dot(self.first_and_second_axis)
        torque_22 = np.cross(center3-self.red_pos,np.array([0,+9.8,0])).dot(self.first_and_second_axis)
        t2 = torque_42 + torque_32 + torque_22

        torque_43 = np.cross(center1-self.green_pos,np.array([0,+9.8,0])).dot(self.first_and_second_axis)
        torque_33 = np.cross(center2-self.green_pos,np.array([0,+9.8,0])).dot(self.first_and_second_axis)
        t3 = torque_33 + torque_43

        torque_44 = np.cross(center1-self.blue_pos,np.array([0,+9.8,0])).dot(self.ee_local_y_axis)
        t4 = torque_44

        return np.matrix([0, t2, t3, t4]).T

    def ts_pd_grav(self, current_joint_angles, current_joint_velocities, desired_position):
        """ PD control with gravity compensation.
            Implementation of the equations on the lecture slides on pd control with
            gravity compensation."""
        P = np.array([280,280,280])
        D = np.array([40,40,40])
        J = self.Jacobian(current_joint_angles)
        xd = J*np.matrix(current_joint_velocities).T
        curr_pos = self.FK(current_joint_angles)
        grav_torques = self.grav(current_joint_angles)
        proportional = np.diag(P)*np.matrix(desired_position-curr_pos).T
        derivative = np.diag(D)*xd
        return J.T*(proportional - derivative) + grav_torques

    def angle_normalize(self, x):
        """Normalizes the angle between pi and -pi"""
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    def angle_normalize_array(self,xs):
        '''Normalizes each angle in xs. '''
        for i in range(len(xs)):
            xs[i] = self.angle_normalize(xs[i])
        return xs

    def coordinate_convert(self, pixels):
        """Converts pixes to meters"""
        return np.array([(pixels[0] - self.env.viewerSize / 2) / self.env.resolution, -(pixels[1] - self.env.viewerSize / 2) / self.env.resolution])

    def go(self):  # The robot has several simulated modes: #These modes are listed in the following format: #Identifier(control mode): Description: Input structure into step
        #function

        # POS: A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles: env.step((np.zeros(3), np.zeros(3), desired joint angles, np.zeros(3)))
        # POS - IMG: Same control as POS, however you must provide the current joint angles and velocities: env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        # VEL: A joint space velocity control, the inputs require the joint angle error and joint velocities: env.step((joint angle error(velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        # TORQUE: Provides direct access to the torque control on the robot: env.step((np.zeros(3), np.zeros(3), np.zeros(3), desired joint torques))

        self.env.controlMode = "VEL"
        self.previousJointAngles = np.zeros(4)
        self.currentJointAngles = np.zeros(4)  # Run 100000 iterations

        # The change in time between iterations can be found in the self.env.dt variable

        # Preamble necessary for VEL mode (doesn't influence TORQUE mode).
        previousJointTorques = np.zeros(4)
        self.prevTime = time.time()
        self.sw = 0
        (xy, xz) = self.env.render()
        xy_valid_target = self.detect_valid_target_new(xy)
        xz_valid_target = self.detect_valid_target_new(xz)
        self.target_prev = np.array([xy_valid_target[0],xy_valid_target[1],xz_valid_target[1]])
        jointAngles = np.array([0, 0, 0, 0])
        for i in range(1000000):
            # self.env.render returns two rgb images facing the xy and xz planes, they are orthogonal
            (xy, xz) = self.env.render()
            dt = self.env.dt
            # ------------------------------------------------------------------
            # Uncomment for VEL mode with no de-stucker.

            # During the first 50 iterations, alter the angles so the manipulator
            # will not start in VEL mode inside a singularity (the dampened
            # squares method's purpose is to avoid singularities).
            # It was either this or modifying how the links are spawned.

            # The de-stucker is prefered as it always (eventually) reaches the
            '''
            if(i > 50):
                xy_valid_target = self.detect_valid_target_new(xy)
                xz_valid_target = self.detect_valid_target_new(xz)

                self.calculate_angles_and_velocities(xy,xz)
                self.env.step((self.IK(self.currentJointAngles,np.array([xy_valid_target[0],xy_valid_target[1],xz_valid_target[1]])), self.currentJointVelocities, np.zeros(3), np.zeros(3)))
            else:
                self.env.controlMode = "POS"
                self.env.step((np.zeros(3),np.zeros(3),np.array([0.3,0.5,1.5,0.1]),np.zeros(3)))
                if(i == 50):
                    self.env.controlMode = "VEL"
            '''

            # ------------------------------------------------------------------
            # Uncomment for VEL mode (with de-stucker for the position the
            # manipulator sometimes gets stuck in).
            xy_valid_target = self.detect_valid_target_new(xy)
            xz_valid_target = self.detect_valid_target_new(xz)
            target_current = np.array([xy_valid_target[0],xy_valid_target[1],xz_valid_target[1]])
            # This part prevents the manipulator from starting in "VEL" mode in a
            # singularity.
            if(i > 35):
                # Basically, what the de-stucker does is that if the target is not
                # reached in 60 seconds then it assumes that the manipulator is stuck.

                # Next, it reverts the angle of the first link and waits until 40
                # seconds have passed (to stabilize it).

                # After 40 seconds have passed, the manipulator enters "VEL" mode
                # again and from our experiments, reaches the target without getting
                # stuck again.

                # The manipulator would get stuck in the same configuration no matter
                # what singularity avoidance method we used. The TA said that this
                # might be due to a bug in the environment.
                if(self.sw == 0):
                    if(np.max(np.absolute(target_current-self.target_prev)) >= 0.001):
                        self.target_prev = target_current
                        self.prevTime = time.time()

                    if(np.max(np.absolute(target_current-self.target_prev)) < 0.001 and time.time()-self.prevTime > 60):
                        self.env.controlMode = "POS"
                        self.currentJointAngles[0] += np.pi
                        self.currentJointAngles = self.angle_normalize_array(self.currentJointAngles)
                        self.env.step((np.zeros(3),np.zeros(3),self.currentJointAngles,np.zeros(3)))
                        self.sw = 1
                        self.count_20 = time.time()
                        print('DE-STUCKER mode enabled.')
                    else:
                        self.calculate_angles_and_velocities(xy,xz)
                        self.env.step((self.IK(self.currentJointAngles,np.array([xy_valid_target[0],xy_valid_target[1],xz_valid_target[1]])), self.currentJointVelocities, np.zeros(3), np.zeros(3)))

                else:
                    if(time.time()-self.count_20 > 40):
                        self.env.controlMode = "VEL"
                        self.sw = 0
                        self.prevTime = time.time()
                        print('DE-STUCKER mode disabled.')
                        self.env.step((self.IK(self.currentJointAngles,np.array([xy_valid_target[0],xy_valid_target[1],xz_valid_target[1]])), self.currentJointVelocities, np.zeros(3), np.zeros(3)))

                    elif(np.max(np.absolute(target_current-self.target_prev)) >= 0.001):
                        self.target_prev = target_current
                        self.prevTime = time.time()
                        self.env.controlMode = "VEL"
                        self.sw = 0
                        print('DE-STUCKER mode disabled.')

                    else:
                        self.env.step((np.zeros(3), np.zeros(3), self.currentJointAngles, np.zeros(3)))
            else:
                self.env.controlMode = "POS"
                self.env.step((np.zeros(3),np.zeros(3),np.array([0.8,0.5,1.5,0.1]),np.zeros(3)))
                if(i == 35):
                    self.env.controlMode = "VEL"


            '''#-------------------------------------------------------------------
            # Uncomment for TORQUE mode with gravity compensation.
            self.env.enable_gravity(True)
            xy_target = self.detect_valid_target_new(xy)
            xz_target = self.detect_valid_target_new(xz)
            self.calculate_angles_and_velocities(xy,xz)
            ee_target = np.array([xy_target[0], xy_target[1], xz_target[1]])
            desired_joint_torques = self.ts_pd_grav(self.currentJointAngles, self.currentJointVelocities, ee_target)
            self.env.step((np.zeros(4), np.zeros(4), np.zeros(4), desired_joint_torques))'''

# main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
