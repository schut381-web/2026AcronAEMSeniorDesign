import jsbsim as jsb
from .state_json import state_json
from .autopilot import Autopilot, FPS_TO_KT
from simcore.shared_state import controls, controls_lock

JSBSIM_AIRCRAFT = "AAM_Vehicle"

class PhysicsEngine:
    def __init__(self):
        self.fdm = jsb.FGFDMExec('.')
        self.fdm.load_model(JSBSIM_AIRCRAFT)
        self.fdm.set_dt(1/120)
        self.dt = 1/120
        self.autopilot = Autopilot(self.dt)     # flies the axes that have a bug set

        self.set_initial_conditions()
        initialized = self.fdm.run_ic()

        if not initialized:
            raise RuntimeError("JSBSim failed to initialize initial conditions")
    
    def apply_controls(self):
        # Craft's current state, used by the autopilot
        meas = {
            "fwd_kt": self.fdm['velocities/u-fps'] * FPS_TO_KT,    # signed forward speed (kt)
            "side_kt": self.fdm['velocities/v-fps'] * FPS_TO_KT,   # sideways drift, right = positive (kt)
            "alt_ft": self.fdm['position/h-agl-ft'],                # same altitude the ALT tape shows
            "vz_fps": -self.fdm['velocities/v-down-fps'],           # climb rate, up = positive
            "hdg_deg": self.fdm['attitude/psi-deg'],
            "theta_rad": self.fdm['attitude/theta-rad'],
            "phi_rad": self.fdm['attitude/phi-rad'],
        }

        with controls_lock:
            cmds = {k: controls[k] for k in ("roll_cmd", "pitch_cmd", "yaw_rate_cmd", "heave_cmd")}
            bugs = {k: controls.get(k) for k in ("airspeed_bug", "altitude_bug", "heading_bug")}

            # Axes with a bug set are flown by the autopilot, the rest stay on the keyboard.
            # Writing the result back into controls means the keys pick up from where the
            # autopilot left off when a bug is turned off (no jump).
            cmds.update(self.autopilot.update(cmds, bugs, meas))
            controls.update(cmds)

        roll_cmd = cmds["roll_cmd"]
        pitch_cmd = cmds["pitch_cmd"]
        yaw_cmd = cmds["yaw_rate_cmd"]
        heave_cmd = cmds["heave_cmd"]

        self.fdm['fcs/refPhi_rad'] = roll_cmd
        self.fdm['fcs/refTheta_rad'] = pitch_cmd
        self.fdm['fcs/rudder-cmd-norm'] = yaw_cmd
        self.fdm['fcs/cmdHeave_nd'] = heave_cmd

    def step(self):
        self.fdm.run()

    def set_initial_conditions(self):
        # Set initial conditions
        self.fdm['ic/velocities-vt-cmd-mps'] = 0.0      #True airspeed (m/s)
        self.fdm['ic/latitude-geod-deg'] = 44.725801    #Geodetic latitude (deg)
        self.fdm['ic/longitude-deg'] = -93.075866       #Longitude (deg)
        self.fdm['ic/altitude-ft'] = 3.5                #Altitude (ft)
        self.fdm['ic/terrain-elevation-ft'] = 0.0       #Terrain elevation (ft)
        self.fdm['ic/psi-deg'] = 90.0                   #Aircraft heading (deg)

        self.fdm['ic/u-fps'] = 0.0            #Forward velocity (body frame) (ft/s)
        self.fdm['ic/v-fps'] = 0.0            #Horizontal velocty (body frame) (ft/s)
        self.fdm['ic/w-fps'] = 0.0            #Vertical velocity (body frame) (ft/s)         
        self.fdm['ic/p-rad_sec'] = 0.0        #Roll rate (rad/s)
        self.fdm['ic/q-rad_sec'] = 0.0        #Pitch rate (rad/s)
        self.fdm['ic/r-rad_sec'] = 0.0        #Yaw rate (rad/s)

        self.fdm['fcs/ScasEngage'] = 1.0      #Activates flight controller

        self.fdm['fcs/refPhi_rad'] = 0.0           #Reference roll angle (rad)
        self.fdm['fcs/refTheta_rad'] = 0.0         #Reference pitch angle (rad)
        self.fdm['fcs/rudder-cmd-norm'] = 0.0      #Rudder command setting
        self.fdm['fcs/cmdHeave_nd'] = 0.0          #Heave command setting


    # TO-DO-STATEJSON
    # Finish the state json class
    def get_state_json(self):
        """
        Returns a dictionary representation of the current simulation state.
        """
        # Create a state_json object and immediately convert to dict
        return state_json(self.fdm).to_dict()