# USER INPUT FILE FOR KEYBOARD ONLY.
# If trying to use controller, name the other file user_input and rename this one something else


import math
from threading import Lock
from pynput import keyboard
from simcore.shared_state import controls, controls_lock

class UserInput:


    ROLL_MIN  = -20.0 / 57.2958     #Radians
    ROLL_MAX  =  20.0 / 57.2958     #Radians
    PITCH_MIN = -20.0 / 57.2958     #Radians
    PITCH_MAX =  20.0 / 57.2958     #Radians
    YAW_RATE_MIN = -0.24    # about 10 deg/s. Faster yaw costs this craft a lot of altitude
    YAW_RATE_MAX = 0.24     # (each J/L press is about 0.9 deg/s)
    HEAVE_MIN = 0.0
    HEAVE_MAX = 1.0

    ANGLE_STEP = 1 / 57.2958    #Radians
    RATE_STEP  = 0.02

    AIRSPEED_BUG_MIN = 0        # kt
    AIRSPEED_BUG_MAX = 999      # kt

    ALTITUDE_BUG_MIN = 0        # ft
    ALTITUDE_BUG_MAX = 5000     # ft

    def start_listener(self):
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            self.apply_user_input(key.char)
        except AttributeError:
            pass

    def apply_user_input(self, key: str):

        with controls_lock:
            if key == 'd':
                controls["roll_cmd"] = max(controls["roll_cmd"] - self.ANGLE_STEP, self.ROLL_MIN)
            elif key == 'a':
                controls["roll_cmd"] = min(controls["roll_cmd"] + self.ANGLE_STEP, self.ROLL_MAX)
            elif key == 's':
                controls["pitch_cmd"] = min(controls["pitch_cmd"] + self.ANGLE_STEP, self.PITCH_MAX)
            elif key == 'w':
                controls["pitch_cmd"] = max(controls["pitch_cmd"] - self.ANGLE_STEP, self.PITCH_MIN)
            elif key == 'l':
                controls["yaw_rate_cmd"] = max(controls["yaw_rate_cmd"] - self.RATE_STEP, self.YAW_RATE_MIN)
            elif key == 'j':
                controls["yaw_rate_cmd"] = min(controls["yaw_rate_cmd"] + self.RATE_STEP, self.YAW_RATE_MAX)
            elif key == 'i':
                controls["heave_cmd"] = min(controls["heave_cmd"] + self.RATE_STEP, self.HEAVE_MAX)
            elif key == 'k':
                controls["heave_cmd"] = max(controls["heave_cmd"] - self.RATE_STEP, self.HEAVE_MIN)
            elif key in ('e', 'E'):
                # Toggle airspeed bug: if it's showing, clear it (NaN = hidden).
                # Otherwise sync it to the craft's actual airspeed (kt).
                # "actual_airspeed_kt" is published by CockpitWindow.handleUpdate().
                current_bug = controls.get("airspeed_bug")
                if current_bug is not None and math.isfinite(current_bug):
                    controls["airspeed_bug"] = float("nan")
                else:
                    actual = controls.get("actual_airspeed_kt")
                    if actual is not None and math.isfinite(actual):
                        controls["airspeed_bug"] = max(self.AIRSPEED_BUG_MIN,
                                                       min(self.AIRSPEED_BUG_MAX, int(round(actual))))
            elif key in ('r', 'R'):
                # Toggle altitude bug: if it's showing, clear it (NaN = hidden).
                # Otherwise sync it to the craft's actual altitude (ft).
                # "actual_altitude_ft" is published by CockpitWindow.handleUpdate().
                current_bug = controls.get("altitude_bug")
                if current_bug is not None and math.isfinite(current_bug):
                    controls["altitude_bug"] = float("nan")
                else:
                    actual = controls.get("actual_altitude_ft")
                    if actual is not None and math.isfinite(actual):
                        controls["altitude_bug"] = max(self.ALTITUDE_BUG_MIN,
                                                       min(self.ALTITUDE_BUG_MAX, int(round(actual))))
            elif key in ('y', 'Y'):
                # Toggle the flight director (F/D) cue on the PFD.
                controls["fd_on"] = not controls.get("fd_on", False)
            elif key in ('h', 'H'):
                # Toggle heading bug: if it's showing, clear it (NaN = hidden).
                # Otherwise sync it to the craft's actual heading (deg).
                # "actual_heading_deg" is published by CockpitWindow.handleUpdate().
                current_bug = controls.get("heading_bug")
                if current_bug is not None and math.isfinite(current_bug):
                    controls["heading_bug"] = float("nan")
                else:
                    actual = controls.get("actual_heading_deg")
                    if actual is not None and math.isfinite(actual):
                        controls["heading_bug"] = int(round(actual)) % 360