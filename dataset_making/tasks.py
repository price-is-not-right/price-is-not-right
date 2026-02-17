"""
Tasks
"""
import numpy as np
from dataset_making.utils import cap, to_osc_pose

np.set_printoptions(linewidth=200, precision=4, suppress=True)

class TaskOperation:
    """Base class for PDDL-driven operations, with common motion loops."""
    def __init__(self, env, detector, randomize=False, noise_std=0.03):
        self.env = env
        self.detector = detector
        self.randomize = randomize
        self.noise_std = noise_std
        # Align our gripper reference with the detector's mapping so XY centering
        # uses the SAME frame the detector uses for `over(gripper,·)`.
        grip_name = detector.object_id.get('gripper', 'gripper0_eef')
        self._grip_name = grip_name
        self._gripper_is_site = False
        self._gripper_site = None
        self._gripper_body = None
        # Try site first
        try:
            self._gripper_site = env.sim.model.site_name2id(grip_name)
            self._gripper_is_site = True
        except Exception:
            # Fallback to body
            self._gripper_body = env.sim.model.body_name2id(grip_name)
            self._gripper_is_site = False
        self._open_sign = -1.0
        # -1 means “holding open,” +1 means “holding closed,” 0 means “no command”
        self._grip_cmd = 0
    def _gripper_pos(self):
        if self._gripper_is_site:
            return np.array(self.env.sim.data.site_xpos[self._gripper_site])
        else:
            return np.array(self.env.sim.data.body_xpos[self._gripper_body])

    
    def _measure_aperture(self):
        j1 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint1'))
        j2 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint2'))
        return abs(j1 - j2)


    def _calibrate_open_sign(self, steps: int = 20):
        """Try +1 and -1 for `steps` frames and see which increases aperture more."""
        base = self._measure_aperture()
        deltas = {}
        for sign in (+1, -1):
            # apply `sign` for a few steps
            for _ in range(steps):
                self.env.step(to_osc_pose(np.array([0, 0, 0, sign])))
            aper = self._measure_aperture()
            deltas[sign] = aper - base
            for _ in range(steps):
                self.env.step(to_osc_pose(np.array([0, 0, 0, -sign])))
        best = max(deltas, key=deltas.get)
        # print(f"[Calibration] open_sign={best}, deltas={deltas}")
        return best


    def record(self, obs, action):
        if hasattr(self.env, 'record_step'):
            self.env.record_step(obs, action)


    def _loop(self, predicate, action_fn, max_steps, render=False):
        """
        Generic loop: until predicate(obs) is True or max_steps reached.
        action_fn(obs) -> raw 4D action
        predicate() checks detector grounding for success.
        """
        for step in range(max_steps):
            obs = self.env.env._get_observations()
            if predicate():
                return True, obs
            raw_action = action_fn(obs)
            raw_action[3] = self._grip_cmd
            action = to_osc_pose(raw_action)
            self.record(obs, action)
            # print(f"full action → {action}")
            obs, *_ = self.env.step(action)
            if render:
                self.env.render()
        return False, obs
    

    def _ascend(self, target_z, speed=10.0, max_steps=500):
        """Move the gripper vertically until z >= target_z."""
        def predicate():
            return float(self._gripper_pos()[2]) >= target_z
        
        def action_fn(obs):
            current_z = float(self._gripper_pos()[2])
            dz = max(0.0, float(target_z - current_z))
            dz = cap(np.array([dz]))[0]
            move = np.array([0.0, 0.0, dz, 0.0])
            if self.randomize and dz > 0.0:
                sigma = self.noise_std * dz
                z_noise = float(np.random.normal(0.0, sigma))
                move[2] += z_noise
            # ensure we don't command downward in ascend
            if move[2] < 0.0:
                move[2] = 0.0
            move[3] = self._grip_cmd
            return move * speed
        
        return self._loop(predicate, action_fn, max_steps)


    def _descend(self, target_z, speed=7.5, max_steps=500,
                 track_body_id=None, track_obj_name=None,
                 xy_k=4.0, xy_deadband=0.0015, xy_max=0.004):
        """
        SIMPLE controller: each step, command XY directly toward the target
        (same style as `_move_xy`) while moving Z straight down toward `target_z`.
        No fancy gains; we just boost XY a bit to fight drift during contact.
        Random XY noise is **disabled** here for stability.
        """
        XY_BOOST = 1  # slightly stronger XY than Z for robustness
        step = {"i": 0}

        def predicate():
            return float(self._gripper_pos()[2]) <= target_z

        def action_fn(obs):
            pos3 = self._gripper_pos()
            current_z = float(pos3[2])

            # Z: strictly downward toward target_z (no upward motion)
            dz = max(0.0, current_z - float(target_z))
            dz = cap(np.array([dz]))[0]
            move = np.array([0.0, 0.0, -dz, 0.0])

            # XY: same style as _move_xy (no noise). If a body to track is given,
            # use its live pose; if it's a peg, prefer peg center map.
            if track_body_id is not None and track_obj_name is not None:
                if hasattr(self.detector, 'peg_target_positions') and track_obj_name in self.detector.peg_target_positions:
                    tgt_xy = np.array(self.detector.peg_target_positions[track_obj_name])
                    tgt3 = np.array([tgt_xy[0], tgt_xy[1], pos3[2]])
                else:
                    tgt3 = np.array(self.env.sim.data.body_xpos[track_body_id])
                dist_xy = tgt3[:2] - pos3[:2]
                dist_xy = cap(dist_xy)
                dist_xy *= XY_BOOST
                move[0], move[1] = dist_xy[0], dist_xy[1]
            else:
                dist_xy = np.zeros(2, dtype=float)

            # Keep strictly downward
            if move[2] > 0.0:
                move[2] = 0.0
            move[3] = self._grip_cmd

            # Debug every 20 steps and near target
            # if (step["i"] % 20 == 0) or dz < 1e-3:
            #     try:
            #         over_dist = float(self.detector.over('gripper', track_obj_name, return_distance=True)) if track_obj_name else float(np.linalg.norm(dist_xy))
            #     except Exception:
            #         over_dist = float(np.linalg.norm(dist_xy))
            #     try:
            #         print(f"[_descend] step={step['i']} z={current_z:.4f}->" \
            #               f"{target_z:.4f} dist_xy={over_dist:.6f} pos_xy={pos3[:2]}" \
            #               f" tgt_xy={(tgt3[:2] if 'tgt3' in locals() else np.array([np.nan, np.nan]))}")
            #     except Exception:
            #         pass
            step["i"] += 1
            return move * speed

        return self._loop(predicate, action_fn, max_steps)

    def _descend_xy_until_on(self, pick_str, goal_str, target_z, speed=8.0, max_steps=500,
                              track_body_id=None, track_obj_name=None,
                              xy_k=3.0, xy_deadband=0.0015, xy_max=0.0035):
        """
        SIMPLE controller: descend while commanding XY directly toward target
        center (same logic as `_move_xy`). XY is boosted to resist drift; noise
        is disabled for stability. Stop when detector reports on(pick, goal).
        """
        XY_BOOST = 2.0
        step = {"i": 0}

        def predicate():
            st = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            return bool(st.get(f"on({pick_str},{goal_str})", False))

        def action_fn(obs):
            pos3 = self._gripper_pos()
            current_z = float(pos3[2])

            dz = max(0.0, current_z - float(target_z))
            dz = cap(np.array([dz]))[0]
            move = np.array([0.0, 0.0, -dz, 0.0])

            if track_body_id is not None and track_obj_name is not None:
                if hasattr(self.detector, 'peg_target_positions') and track_obj_name in self.detector.peg_target_positions:
                    tgt_xy = np.array(self.detector.peg_target_positions[track_obj_name])
                    tgt3 = np.array([tgt_xy[0], tgt_xy[1], pos3[2]])
                else:
                    tgt3 = np.array(self.env.sim.data.body_xpos[track_body_id])
                dist_xy = tgt3[:2] - pos3[:2]
                dist_xy = cap(dist_xy)
                dist_xy *= XY_BOOST
                move[0], move[1] = dist_xy[0], dist_xy[1]
            else:
                dist_xy = np.zeros(2, dtype=float)

            if move[2] > 0.0:
                move[2] = 0.0
            move[3] = self._grip_cmd

            # if (step["i"] % 20 == 0) or dz < 1e-3:
            #     try:
            #         on_flag = bool(self.detector.on(pick_str, goal_str))
            #     except Exception:
            #         on_flag = False
            #     try:
            #         print(f"[_descend_xy_until_on] step={step['i']} z={current_z:.4f}->" \
            #               f"{target_z:.4f} on={on_flag} pos_xy={pos3[:2]}" \
            #               f" tgt_xy={(tgt3[:2] if 'tgt3' in locals() else np.array([np.nan, np.nan]))}")
            #     except Exception:
            #         pass
            step["i"] += 1
            return move * speed

        return self._loop(predicate, action_fn, max_steps)


    def _move_xy(self, body_id, obj_name, speed=10.0, max_steps=500):
        """
        Center the gripper over a target object in XY plane.
        Handles pegs (use predefined peg center) and objects (use body pos).
        """
        def predicate():
            state = self.detector.get_groundings(
                as_dict=True, binary_to_float=False, return_distance=False
            )
            # print(f"state = {state}")
            return state.get(f"over(gripper,{obj_name})", False)

        def action_fn(obs):
            pos = self._gripper_pos()
            if hasattr(self.detector, 'peg_target_positions') and obj_name in self.detector.peg_target_positions:
                tgt_xy = np.array(self.detector.peg_target_positions[obj_name])
                tgt = np.array([tgt_xy[0], tgt_xy[1], pos[2]])
            else:
                tgt = np.array(self.env.sim.data.body_xpos[body_id])
            dist_xy = tgt[:2] - pos[:2]
            dist_xy = cap(dist_xy)
            move = np.array([dist_xy[0], dist_xy[1], 0.0, 0.0])
            if self.randomize:
                sigma = self.noise_std * float(np.linalg.norm(dist_xy))
                if sigma > 0.0:
                    noise_xy = np.random.normal(0.0, sigma, size=2)
                    move[:2] += noise_xy
            move[3] = self._grip_cmd
            return move * speed

        return self._loop(predicate, action_fn, max_steps)

    def _move_xy_object(self, body_id, obj_name, speed=10.0, max_steps=500):
        """Center the gripper over an OBJECT in XY using the object's live body pose.
        Ignores any peg target map to guarantee we're not using a peg center.
        Prints detailed debug every ~20 steps and near convergence.
        """
        step = {"i": 0}

        def predicate():
            # Use detector.over distance if available for consistent measurement
            dist = None
            over_flag = False
            try:
                dist = float(self.detector.over('gripper', obj_name, return_distance=True))
                over_flag = bool(self.detector.over('gripper', obj_name))
            except Exception:
                pos = self._gripper_pos()[:2]
                tgt = np.array(self.env.sim.data.body_xpos[body_id])[:2]
                dist = float(np.linalg.norm(tgt - pos))
                # Fallback to state dict boolean
                state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                over_flag = bool(state.get(f"over(gripper,{obj_name})", False))

            # if (step["i"] % 20 == 0) or (dist is not None and dist < 1e-3):
            #     try:
            #         pos = self._gripper_pos()[:2]
            #         tgt = np.array(self.env.sim.data.body_xpos[body_id])[:2]
            #         print(f"[move_xy_object] obj={obj_name} step={step['i']} dist_xy={dist:.6f} over={over_flag} pos={pos} tgt={tgt}")
            #     except Exception:
            #         print(f"[move_xy_object] obj={obj_name} step={step['i']} dist_xy={dist:.6f} over={over_flag}")
            return over_flag

        def action_fn(obs):
            pos3 = self._gripper_pos()
            tgt3 = np.array(self.env.sim.data.body_xpos[body_id])  # force body pos
            dist_xy = tgt3[:2] - pos3[:2]
            dist_xy = cap(dist_xy)
            move = np.array([dist_xy[0], dist_xy[1], 0.0, 0.0])
            if self.randomize:
                sigma = self.noise_std * float(np.linalg.norm(dist_xy))
                if sigma > 0.0:
                    move[:2] += np.random.normal(0.0, sigma, size=2)
            move[3] = self._grip_cmd

            # Debug print every 20 steps
            # if step["i"] % 20 == 0:
            #     print(f"[move_xy_object] action step={step['i']} move_xy={move[:2]} dist_xy={dist_xy} speed={speed}")
            step["i"] += 1
            return move * speed

        return self._loop(predicate, action_fn, max_steps)

    ### For Kinova gripper
    # def _gripper_actuate(self, open_grip=True, max_steps=50):
    #     """Open or close the gripper."""
    #     pred = 'open_gripper' if open_grip else 'grasped'
    #     def predicate():
    #         # Force a dict result so .get(...) works
    #         state = self.detector.get_groundings(
    #             as_dict=True, binary_to_float=False, return_distance=False
    #         )
    #         return state.get(f"{pred}(gripper)", False)
    #     def action_fn(obs):
    #         val = -1 if open_grip else 1
    #         return np.array([0, 0, 0, val])
    #     return self._loop(predicate, action_fn, max_steps)
    
    ### For Panda gripper
    def _gripper_actuate(self, open_grip=True, max_steps=50):
        pred = 'open_gripper' if open_grip else 'grasped'

        def predicate():
            st = self.detector.get_groundings(as_dict=True,
                                              binary_to_float=False,
                                              return_distance=False)
            if open_grip:
                return bool(st.get(f"{pred}(gripper)", False))
            else:
                return any(v for k, v in st.items() if k.startswith("grasped(") and v)

        sign = self._open_sign if open_grip else -self._open_sign
        def action_fn(obs):
            return np.array([0, 0, 0, sign])

        self._grip_cmd = sign
        return self._loop(predicate, action_fn, max_steps)


    def _lift(self, height=0.4, speed=10.0, max_steps=300):
        """Lift the object after grasping."""
        return self._ascend(target_z=self._gripper_pos()[2] + height,
                                   speed=speed, max_steps=max_steps)
    

    def _descend_until_on(self, pick_str, goal_str, target_z, speed=10.0, max_steps=200):
        def predicate():
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            return state.get(f'on({pick_str},{goal_str})', False)
        def action_fn(obs):
            gripper_z = float(self._gripper_pos()[2])
            dz = max(0.0, float(gripper_z - target_z))
            dz = cap(np.array([dz]))[0]
            move = np.array([0.0, 0.0, -dz, 0.0])
            if self.randomize and dz > 0.0:
                sigma = self.noise_std * dz
                z_noise = float(np.random.normal(0.0, sigma))
                move[2] -= z_noise
            if move[2] > 0.0:
                move[2] = 0.0
            move[3] = self._grip_cmd
            return move * speed
        return self._loop(predicate, action_fn, max_steps)

class PickOperation(TaskOperation):
    """Pick up the specified object."""
    def __init__(self, env, detector, randomize, noise_std, object_id):
        super().__init__(env, detector, randomize, noise_std)
        self.object_id = object_id
        self.body_id = env.sim.model.body_name2id(detector.object_id[object_id])

    def execute(self, obs):
        # print(f"[PickOperation] using open_sign = {self._open_sign}")
        # print(f"[PickOperation] object={self.object_id} body_id={self.body_id} gripper='{self._grip_name}' is_site={self._gripper_is_site}")

        # Compute a dynamic hover height: 10 cm above the object
        object_z = float(self.env.sim.data.body_xpos[self.body_id][2]) + 0.0125
        ref_z = 1.0 #object_z + 0.10

        # 1) Move up to hover
        ok, obs = self._ascend(target_z=ref_z)
        if not ok:
            print(f"[PickOperation] ❌ failed to move above object at z={ref_z}")
            return False, obs

        # 2) Center in XY over the object
        ok, obs = self._move_xy_object(self.body_id, self.object_id)
        if not ok:
            print(f"[PickOperation] ❌ failed to XY-center over {self.object_id}")
            return False, obs

        # 3) Open the gripper
        ok, obs = self._gripper_actuate(open_grip=True)
        if not ok:
            print("[PickOperation] ❌ failed to open gripper")
            return False, obs

        # 4) Descend straight down onto the object
        ok, obs = self._descend(
            target_z=object_z,
            track_body_id=self.body_id,
            track_obj_name=self.object_id,
            speed=5,
            xy_k=4.5,
            xy_deadband=0.0015,
            xy_max=0.0045,
        )
        if not ok:
            print(f"[PickOperation] ❌ failed to descend to z={object_z}")
            return False, obs

        # Debug: measure XY error just before closing
        # try:
        #     dist_xy = float(self.detector.over('gripper', self.object_id, return_distance=True))
        #     gpos = self._gripper_pos()[:2]
        #     cpos = np.array(self.env.sim.data.body_xpos[self.body_id])[:2]
        #     print(f"[PickOperation] pre-close XY error={dist_xy:.6f}  gripper_xy={gpos}  cube_xy={cpos}")
        # except Exception as e:
        #     print(f"[PickOperation] pre-close XY debug failed: {e}")

        # 5) Close the gripper
        ok, obs = self._gripper_actuate(open_grip=False)
        if not ok:
            print("[PickOperation] ❌ failed to close on object")
            return False, obs

        # 7) Keep closing for a few more steps to ensure full closure
        EXTRA_CLOSE_STEPS = 10
        for _ in range(EXTRA_CLOSE_STEPS):
            obs = self.env.env._get_observations()
            action = to_osc_pose(np.array([0, 0, 0, -self._open_sign]))
            self.record(obs, action)
            obs, *_ = self.env.step(action)

        # 6) Lift the object slightly
        ok, obs = self._ascend(target_z=0.2)
        if not ok:
            print("[PickOperation] ❌ failed to lift object")
            return False, obs

        print(f"[PickOperation] ✅ picked {self.object_id}")
        return True, obs
    

class PlaceOperation(TaskOperation):
    """Place a held object at the target location."""
    def __init__(self, env, detector, randomize, noise_std, object_id, placement_id):
        super().__init__(env, detector, randomize, noise_std)
        self._grip_cmd = -self._open_sign
        self.object_id = object_id
        self.placement_id = placement_id 
        self.body_id = env.sim.model.body_name2id(detector.object_id[placement_id])

    def execute(self, obs):
        # Move above placement
        ok, obs = self._ascend(target_z=1.0)
        if not ok: 
            print(f"[PlaceOperation] ❌ failed to ascend to {1.0}")
            return False, obs
        
        ok, obs = self._move_xy(self.body_id, self.placement_id, speed=10, max_steps=1000)
        if not ok: 
            print(f"[PlaceOperation] ❌ failed to move over {self.placement_id}")
            return False, obs
        
        # # Descend to drop
        # ok, obs = self._descend(target_z=place_z)
        # if not ok: 
        #     print(f"[PlaceOperation] ❌ failed to descend to {place_z}")
        #     return False, obs

        place_z = float(self.env.sim.data.body_xpos[self.body_id][2])
        ok, obs = self._descend_xy_until_on(
            pick_str=self.object_id,
            goal_str=self.placement_id,
            target_z=place_z,
            track_body_id=self.body_id,
            track_obj_name=self.placement_id,
            speed=8.0,
            xy_k=3.0,
            xy_deadband=0.0015,
            xy_max=0.0035,
        )
        if not ok:
            print(f"[PlaceOperation] ❌ failed to descend until on({self.object_id},{self.placement_id})")
            return False, obs
        
        # Open to release
        ok, obs = self._gripper_actuate(open_grip=True)
        if not ok: 
            print(f"[PlaceOperation] ❌ failed to ascend to open gripper")
            return False, obs
        
        # Retract
        for _ in range(15):
            obs = self.env.env._get_observations()
            action = to_osc_pose(np.array([0,0,1,0]))
            self.record(obs, action)
            obs, *_ = self.env.step(action)

        print(f"[PlaceOperation] ✅ placed {self.object_id} on {self.placement_id}")
        return True, obs
    

class TurnOnOperation(TaskOperation):
    """Switch a binary button on (e.g., stove)."""
    def __init__(self, env, detector, randomize, noise_std, object_id):
        super().__init__(env, detector, randomize, noise_std)
        self.object_id = object_id
        self.body_id = env.sim.model.body_name2id(detector.object_id[object_id])

    def execute(self, obs):
        # Move above
        ok, obs = self._ascend(target_z=1.1)
        if not ok: return False, obs
        ok, obs = self._move_xy(self.body_id, self.object_id)
        if not ok: return False, obs
        # Descend to switch
        switch_z = float(self.env.sim.data.body_xpos[self.body_id][2])
        ok, obs = self._ascend(target_z=switch_z)
        if not ok: return False, obs
        # Actuate on
        ok, obs = self._gripper_actuate(open_grip=False)
        if not ok: return False, obs
        # Retract
        for _ in range(15):
            obs = self.env.env._get_observations()
            action = to_osc_pose(np.array([0,0,1,0]))
            self.record(obs, action)
            obs, *_ = self.env.step(action)
        return True, obs
    

class TurnOffOperation(TurnOnOperation):
    """Switch a binary button off."""
    def execute(self, obs):
        # Same as TurnOn but open_grip inverted for actuation
        return super().execute(obs)
