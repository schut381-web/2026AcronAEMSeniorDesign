"""
This is the shared state of the sim. Used to pass between files. 

"""

from threading import Lock

# user controls going INTO physics
controls = {
    "roll_cmd": 0.0,            #rad
    "pitch_cmd": 0.0,           #rad
    "yaw_rate_cmd": 0.0,
    "heave_cmd": 0.001,

    # Bugs. A number = the autopilot flies that axis to this value, NaN = bug off (manual keys).
    "airspeed_bug": float("nan"),       # kt
    "altitude_bug": float("nan"),       # ft AGL
    "heading_bug": float("nan"),        # deg

    "fd_on": False,                     # flight director cue shown on the PFD (Y key)

    # Craft's actual values, published by the UI each update (used by the bug sync keys)
    "actual_airspeed_kt": float("nan"),
    "actual_altitude_ft": float("nan"),
    "actual_heading_deg": float("nan"),
}
controls_lock = Lock()

# physics state coming OUT to the UI
latest_state = {}
state_lock = Lock()