"""
Bug-hold autopilot (outer loops).

When a bug in shared_state.controls holds a number, this module flies the
matching axis instead of the keyboard:

    airspeed_bug (kt)  -> pitch_cmd      nose down = faster
    altitude_bug (ft)  -> heave_cmd      collective, AGL like the ALT tape
    heading_bug  (deg) -> yaw_rate_cmd   rudder norm, turns the short way round

NaN / None / missing = bug off, and that axis stays manual. Roll stays manual.

These loops only produce the same commands the keyboard produces. The attitude
and rate loops in FlightControl.xml still do the inner work.

The commands are written back into controls by physics_engine, so turning a
bug on or off never causes a jump (the integrators start from whatever command
was already being flown).
"""

import math

FPS_TO_KT = 0.592484      # same factor state_json uses for TAS

# ============================ TUNING ============================
# Tuned on the AAM_Vehicle JSBSim model (hover heave ~0.85).
# If something hunts, change one value at a time and make small changes.

# Airspeed hold: speed error -> pitch angle command.
SPD_KP = 0.03                           # rad of pitch per kt of speed error
SPD_KI = 0.004                          # rad of pitch per kt of error, per second
SPD_KD = 0.06                           # rad of pitch per (kt/s) of acceleration: damps speed overshoot
SPD_PITCH_LIMIT = math.radians(15.0)    # largest pitch the hold will ask for

# Altitude hold: altitude error -> climb rate -> heave command.
ALT_KP = 0.5                            # (ft/s climb rate) per ft of altitude error
ALT_CLIMB_MAX = 8.0                     # ft/s
ALT_DESCENT_MAX = 5.0                   # ft/s
VS_KP = 0.04                            # heave per ft/s of climb rate error
VS_KI = 0.03                            # heave per (ft/s of error), per second
HEAVE_MIN = 0.0
HEAVE_MAX = 1.0
MIN_TILT_COS = 0.5                      # tilt compensation never divides by less than this

# Heading hold: heading error -> yaw rate -> rudder command.
HDG_KP = 1.0                            # (rad/s yaw rate) per rad of heading error
HDG_KD = 1.0                            # (rad/s yaw rate) taken off per (rad/s) of actual turn rate: damps overshoot
HDG_RATE_MAX = math.radians(10.0)       # rad/s, at hover (same limit as the J/L keys)
TURN_BANK = math.radians(8.0)           # at speed, turn no faster than this bank angle can follow
                                        # (20 kt -> ~7.7 deg/s, 40 kt -> ~3.8 deg/s)
DERIV_TAU = 0.3                         # s: smoothing on the measured turn rate and acceleration
YAW_RATE_PER_NORM = 0.75                # FlightControl.xml: refYaw_rps = 0.75 * rudder-cmd-norm
YAW_SIGN = 1.0                          # set to -1.0 if +rudder turns the craft toward LOWER headings

# Turn coordination (only while BOTH the airspeed and heading bugs are on).
# Banks into turns like an aircraft and cancels sideways drift, so the craft's motion turns with
# the nose instead of sliding sideways. Without this, a big turn at speed can lose control.
COORD_K_SIDE = 0.02                     # rad of bank per kt of sideways drift
COORD_ROLL_LIMIT = math.radians(20.0)   # largest bank it will use (matches the keyboard roll limit)
COORD_ROLL_RATE = math.radians(10.0)    # rad/s: fastest the bank command may change
# ================================================================


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _number(v):
    """Return v as a float, or None if it is missing / NaN / not a number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class Autopilot:
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self._spd_on = False
        self._alt_on = False
        self._hdg_on = False
        self._coord_on = False
        self._spd_i = 0.0       # pitch integrator (rad, positive = nose down)
        self._alt_i = 0.0       # heave integrator (level-flight equivalent)
        self._prev_fwd = None   # for the smoothed forward acceleration (kt/s)
        self._prev_hdg = None   # for the smoothed turn rate (rad/s)
        self._accel = 0.0
        self._turn_rate = 0.0

    def _track_rates(self, fwd, hdg):
        """Smoothed acceleration and turn rate, kept up to date even while the bugs are off."""
        a = self.dt / (DERIV_TAU + self.dt)
        if fwd is not None:
            if self._prev_fwd is not None:
                self._accel += a * ((fwd - self._prev_fwd) / self.dt - self._accel)
            self._prev_fwd = fwd
        if hdg is not None:
            if self._prev_hdg is not None:
                d = (hdg - self._prev_hdg + 540.0) % 360.0 - 180.0      # handles 359 -> 0
                self._turn_rate += a * (math.radians(d) / self.dt - self._turn_rate)
            self._prev_hdg = hdg

    def update(self, cmds, bugs, meas):
        """
        cmds: current commands from controls (roll_cmd, pitch_cmd, yaw_rate_cmd, heave_cmd)
        bugs: airspeed_bug, altitude_bug, heading_bug (NaN / None = off)
        meas: tas_kt, alt_ft, vz_fps (climb positive), hdg_deg, theta_rad, phi_rad
        Returns {command name: value} for the axes the autopilot is flying.
        """
        out = {}
        fwd = _number(meas.get("fwd_kt", meas.get("tas_kt")))
        self._track_rates(fwd, _number(meas.get("hdg_deg")))

        # ---- Airspeed hold -> pitch ----
        # Uses the signed forward speed so backward drift can't be mistaken for forward flight.
        tgt = _number(bugs.get("airspeed_bug"))
        if tgt is not None and fwd is not None:
            out["pitch_cmd"] = self._airspeed_hold(max(0.0, tgt), cmds["pitch_cmd"], fwd)
        else:
            self._spd_on = False

        # ---- Altitude hold -> heave ----
        tgt = _number(bugs.get("altitude_bug"))
        alt = _number(meas.get("alt_ft"))
        vz = _number(meas.get("vz_fps"))
        if tgt is not None and alt is not None and vz is not None:
            theta = _number(meas.get("theta_rad")) or 0.0
            phi = _number(meas.get("phi_rad")) or 0.0
            cos_tilt = max(MIN_TILT_COS, math.cos(theta) * math.cos(phi))
            out["heave_cmd"] = self._altitude_hold(max(0.0, tgt), cmds["heave_cmd"], alt, vz, cos_tilt)
        else:
            self._alt_on = False

        # ---- Heading hold -> yaw rate ----
        tgt = _number(bugs.get("heading_bug"))
        hdg = _number(meas.get("hdg_deg"))
        if tgt is not None and hdg is not None:
            out["yaw_rate_cmd"] = self._heading_hold(tgt, hdg, fwd)
            self._hdg_on = True
        else:
            if self._hdg_on:
                out["yaw_rate_cmd"] = 0.0       # bug just switched off: stop turning, don't keep spinning
            self._hdg_on = False

        # ---- Turn coordination -> roll (only with both speed and heading bugs on) ----
        side = _number(meas.get("side_kt"))
        if "pitch_cmd" in out and "yaw_rate_cmd" in out and self._hdg_on and fwd is not None and side is not None:
            out["roll_cmd"] = self._coordinate(cmds["roll_cmd"], fwd, side)
            self._coord_on = True
        else:
            if self._coord_on:
                out["roll_cmd"] = 0.0           # hand roll back to the keys level
            self._coord_on = False

        return out

    # ------------------------------------------------------------------
    def _airspeed_hold(self, target_kt, pitch_cmd, tas_kt):
        err = target_kt - tas_kt                    # kt, positive = too slow
        p = SPD_KP * err - SPD_KD * self._accel     # ease off as the speed is already building

        if not self._spd_on:                        # start from the pitch already being flown
            self._spd_i = -pitch_cmd - p
            self._spd_on = True

        # u is "nose down" demand. Stop integrating while saturated and still pushing.
        u = p + self._spd_i + SPD_KI * err * self.dt
        if not (abs(u) > SPD_PITCH_LIMIT and u * err > 0):
            self._spd_i += SPD_KI * err * self.dt

        u = _clip(p + self._spd_i, -SPD_PITCH_LIMIT, SPD_PITCH_LIMIT)
        return -u                                   # nose down is negative pitch

    def _altitude_hold(self, target_ft, heave_cmd, alt_ft, vz_fps, cos_tilt):
        err = target_ft - alt_ft                    # ft, positive = too low
        vz_cmd = _clip(ALT_KP * err, -ALT_DESCENT_MAX, ALT_CLIMB_MAX)
        verr = vz_cmd - vz_fps
        p = VS_KP * verr

        if not self._alt_on:                        # start from the heave already being flown
            self._alt_i = heave_cmd * cos_tilt - p
            self._alt_on = True

        # Stop integrating while the output is pinned and the error pushes further into the stop.
        cmd = (p + self._alt_i + VS_KI * verr * self.dt) / cos_tilt
        pinned = (cmd > HEAVE_MAX and verr > 0) or (cmd < HEAVE_MIN and verr < 0)
        if not pinned:
            self._alt_i += VS_KI * verr * self.dt

        # Dividing by cos(tilt) keeps the vertical thrust the same when the craft pitches or rolls.
        return _clip((p + self._alt_i) / cos_tilt, HEAVE_MIN, HEAVE_MAX)

    def _coordinate(self, roll_cmd, fwd_kt, side_kt):
        # Bank needed to turn the flight path at the current turn rate, minus a correction for sideways drift.
        v_fps = max(0.0, fwd_kt) / FPS_TO_KT
        bank = math.atan(v_fps * self._turn_rate / 32.174) - COORD_K_SIDE * side_kt
        bank = _clip(bank, -COORD_ROLL_LIMIT, COORD_ROLL_LIMIT)
        step = COORD_ROLL_RATE * self.dt                # smooth: no sudden bank changes
        return _clip(bank, roll_cmd - step, roll_cmd + step)

    def _heading_hold(self, target_deg, hdg_deg, fwd_kt):
        err_deg = (target_deg - hdg_deg + 540.0) % 360.0 - 180.0    # -180..180, short way round
        # Ease off the turn command as the actual turn rate builds (the inner yaw loop is slow).
        rate = HDG_KP * math.radians(err_deg) - HDG_KD * self._turn_rate
        # Turn slower when moving fast, so the craft's path can follow the nose without sliding sideways.
        v_fps = abs(fwd_kt or 0.0) / FPS_TO_KT
        rate_max = HDG_RATE_MAX if v_fps < 1.0 else min(HDG_RATE_MAX, 32.174 * math.tan(TURN_BANK) / v_fps)
        rate = _clip(rate, -rate_max, rate_max)
        return _clip(YAW_SIGN * rate / YAW_RATE_PER_NORM, -1.0, 1.0)
